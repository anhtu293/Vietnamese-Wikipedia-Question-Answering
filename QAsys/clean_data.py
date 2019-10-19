from gensim.models import KeyedVectors
from pyvi import ViTokenizer
import re
import string
import json
import numpy as np
from preprocess import preprocess_sentence

WORD_VECTOR_DIM = 400
MAX_SENTENCE_LENGTH = 100
ZERO = np.zeros(WORD_VECTOR_DIM)
word2vec = KeyedVectors.load_word2vec_format('baomoi.model.bin', binary=True)

def make_w2vec_matrix(question, paragraph, model):
	train_sentence = preprocess_sentence(question) + ' | ' + preprocess_sentence(paragraph)
	tokens = ViTokenizer.tokenize(train_sentence).split()
	vectors = []
	for token in tokens:
		if token in model:
			vectors.append(model[token])
		else:
			vectors.append(model['unknown'])
	# padding
	if len(vectors) > MAX_SENTENCE_LENGTH:
		vectors = vectors[:MAX_SENTENCE_LENGTH]
	elif len(vectors) < MAX_SENTENCE_LENGTH:
		for i in range(MAX_SENTENCE_LENGTH - len(vectors)):
			vectors.append(ZERO)
	return np.array(vectors)

def read_train_data(filepath):
	#Read data
	with open(filepath) as file:
		data = json.load(file)
	#Process data
	X = []
	y = []
	for case in data:
		#Process X
		X.append(make_w2vec_matrix(case['question'], case['text'], word2vec))
		#Process y
		y.append(1 if case['label'] == True else 0)
	# return np.array(X), np.array(y)
	return np.array(X), np.array(y)

def read_test_data(filepath):
	#Read data
	with open(filepath) as file:
		data = json.load(file)
	#Process data
	for case in data:
		for paragraph in case['paragraphs']:
			x = make_w2vec_matrix(case['question'], paragraph['text'], word2vec)
			#Reshape to 3 dimensions to fit keras input dim
			paragraph['x'] = np.expand_dims(x, axis = 0)
	return data

if __name__ == "__main__":
	X, y = read_train_data('train.json')
	print(X.shape)