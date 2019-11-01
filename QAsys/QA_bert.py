# ./QAsys/QA_model.py
# First model for Question Answering system.
# This model is built with LSTM
#
# 1) This model consists of 2 inputs : embeddings vectors of question and embeddings vector of answer.
# 2) Inputs will be passed bidirectional LSTM independently to create 2 different distributed
#   representation of question and answer.
# 3) Next hidden layers try to know whether question and answer are in same context.
# 4) Output :  Whether this answer is suitable for the question

import os
import sys
import random

sys.path.append("./bert")
import optimization
import modeling
import tokenization
import json
from run_classifier import InputExample, input_fn_builder, convert_examples_to_features
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from preprocess import preprocess_sentence
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling1D, Input, Dense, Multiply, MaxPooling1D, LSTM, Conv1D, \
    Bidirectional, concatenate, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from clean_data import read_train_data, make_w2vec_matrix
from time import time

def get_args():
    flags = tf.flags
    FLAGS = flags.FLAGS
    ## Required parameters
    flags.DEFINE_string(
        "data_dir", None,
        "The input data dir. Should contain the .json files (or other data files) "
        "for the task.")

    flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_string("vocab_file", None,
                        "The vocabulary file that the BERT model was trained on.")

    flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

    ## Other parameters

    flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

    flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

    flags.DEFINE_bool("do_train", True, "Whether to run training.")

    flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

    flags.DEFINE_bool(
        "do_predict", False,
        "Whether to run the model in inference mode on the test set.")

    flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

    flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

    flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

    flags.DEFINE_float("num_train_epochs", 3.0,
                       "Total number of training epochs to perform.")

    flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

    flags.DEFINE_integer("save_checkpoints_steps", 1000,
                         "How often to save the model checkpoint.")

    flags.DEFINE_integer("iterations_per_loop", 1000,
                         "How many steps to make in each estimator call.")


    return FLAGS

class WikiProcessor(object):
    def _load_json(self, filepath):
        with open(filepath) as file:
            data = json.load(file)
        return data

    def get_train_examples(self, data_path):
        return self._create_examples(self._load_json(os.path.join(data_path, "train.json")), "train")

    def get_test_examples(self, data_path):
        return self._create_examples(self._load_json(os.path.join(data_path, "test.json")), "test")

    def get_labels(self):
        return([0, 1])

    def _create_examples(self, file, type_set):
        examples = []
        if type_set == "train":
            for row in file:
                guid = row["id"]
                tokens_question = preprocess_sentence(row["question"])
                tokens_answer = preprocess_sentence(row["text"])
                label = row["label"]
                examples.append(
                    InputExample(guid = guid, text_a = tokens_question, text_b = tokens_question, label = label)
                )
        else:
            for row in file:
                guid = row["__id__"]
                tokens_question = preprocess_sentence(row["question"])
                for paragraph in row["paragraphs"]:
                    tokens_answer = preprocess_sentence(row["text"])
                    examples.append(
                        InputExample(guid = guid, text_a = tokens_question, text_b = tokens_question, label = label)
                    )
        return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings = False)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu = False)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


if __name__ == '__main__':
    FLAGS = get_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    processor = WikiProcessor()


    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    #tf.gfile.MakeDirs(FLAGS.output_dir)

    label_list = processor.get_labels()

    #tokenizer for input_fn
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file)

    #config
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    run_config = tf.contrib.tpu.RunConfig(
        model_dir = FLAGS.output_dir,
        save_checkpoints_steps = FLAGS.save_checkpoints_steps)

    #parameters for training
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config = bert_config,
        num_labels = len(label_list),
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = num_train_steps,
        num_warmup_steps = num_warmup_steps)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = False,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.train_batch_size,
        eval_batch_size = FLAGS.eval_batch_size,
        predict_batch_size = FLAGS.predict_batch_size)

    #split data
    train_loads = processor.get_train_examples(FLAGS.data_dir)
    #train_loads = random.shuffle(train_loads)
    train_data = train_loads[:10000]
    eval_data = train_loads[10000:]

    #TRAINING
    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_features = convert_examples_to_features(train_data, label_list, FLAGS.max_seq_length, tokenizer)
        train_input_fn = input_fn_builder(
            features = train_features,
            seq_length = FLAGS.max_seq_length,
            is_training = True,
            drop_remainder = False)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    #EVALUATING
    if FLAGS.do_eval:
        eval_features = convert_examples_to_features(eval_data, label_list, FLAGS.max_seq_length, tokenizer)
        num_actual_eval_examples = len(eval_features)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_input_fn = input_fn_builder(
            features = eval_features,
            seq_length = FLAGS.max_seq_length,
            is_training = False,
            drop_remainder = False)
        result = estimator.evaluate(input_fn = eval_input_fn, steps = None)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        test_data = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        test_features = convert_examples_to_features(test_data, label_list, FLAGS.max_seq_length, tokenizer)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(test_data), num_actual_predict_examples,
                        len(test_data) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_input_fn = input_fn_builder(
            features = test_features,
            seq_length = FLAGS.max_seq_length,
            is_training = False,
            drop_remainder = False)
        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

