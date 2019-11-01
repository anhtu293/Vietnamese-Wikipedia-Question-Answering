from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from preprocess import preprocess_sentence

import re
import string
import json
import numpy as np
import sys

MIN_LENGTH_QUESTION = 20
MIN_LENGTH_ANSWER = 64
WORD_VECTOR_DIM = 400
MAX_SENTENCE_LENGTH = 100
ZERO = np.zeros(WORD_VECTOR_DIM)
word2vec = KeyedVectors.load_word2vec_format('../backbones/baomoi.model.bin', binary=True)


def make_w2vec_matrix(question, paragraph, model=word2vec):
    train_question = preprocess_sentence(question)
    train_answers = preprocess_sentence(paragraph)
    tokens_question = ViTokenizer.tokenize(train_question).split()
    tokens_answer = ViTokenizer.tokenize(train_answers).split()
    question_embs = []
    answer_embs = []
    for i in range(len(tokens_question)):
        if tokens_question[i] in model:
            question_embs.append(model[tokens_question[i]])
        else:
            question_embs.append(model['unknown'])
    for i in range(len(tokens_answer)):
        if tokens_answer[i] in model:
            answer_embs.append(model[tokens_answer[i]])
        else:
            answer_embs.append(model['unknown'])
    question_embs = np.array(question_embs)
    answer_embs = np.array(answer_embs)

    """
	if question_embs.shape[0] < MIN_LENGTH_QUESTION:
	question_embs = np.pad(question_embs, ((4,4), (0,0)))
	"""

    if answer_embs.shape[0] < MIN_LENGTH_ANSWER:
        paddings = np.ceil(MIN_LENGTH_ANSWER / answer_embs.shape[0])
        d = np.copy(answer_embs)
        for i in range(int(paddings)):
            answer_embs = np.concatenate((answer_embs, d))

    return question_embs, answer_embs


def read_train_data(filepath):
    # Read data
    with open(filepath) as file:
        data = json.load(file)
    # Process data
    X = []
    y = []
    for case in data:
        # Process X
        X.append([case['question'], case['text']])
        # Process y
        y.append(1 if case['label'] == True else 0)
    # return np.array(X), np.array(y)
    return np.array(X), np.array(y)


def read_test_data(filepath):
    # Read data
    with open(filepath) as file:
        data = json.load(file)
    # Process data
    for case in data:
        for paragraph in case['paragraphs']:
            x = make_w2vec_matrix(case['question'], paragraph['text'], word2vec)
            # Reshape to 3 dimensions to fit keras input dim
            paragraph['x'] = np.expand_dims(x, axis=0)
    return data


if __name__ == "__main__":
    X, y = read_train_data('train.json')
    print(X.shape)
