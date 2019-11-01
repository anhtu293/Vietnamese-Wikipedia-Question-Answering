export BERT_BASE_DIR=../backbones/bert # or multilingual_L-12_H-768_A-12
export DATA_DIR=./data/

python3 QA_bert.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=./checkpoints