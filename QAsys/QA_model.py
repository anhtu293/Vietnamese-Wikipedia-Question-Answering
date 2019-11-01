#./QAsys/QA_model.py
#First model for Question Answering system. 
#This model is built with LSTM
#
#1) This model consists of 2 inputs : embeddings vectors of question and embeddings vector of answer.
#2) Inputs will be passed bidirectional LSTM independently to create 2 different distributed 
#   representation of question and answer.
#3) Next hidden layers try to know whether question and answer are in same context. 
#4) Output :  Whether this answer is suitable for the question

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, LSTM, Conv1D, Bidirectional, concatenate, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from clean_data import read_train_data, make_w2vec_matrix
import os
import time
from datetime import datetime

class QA_selection:
    def __init__(self, num_features, question_length, answer_length, question_encoder_shape, answer_encoder_shape,batch_size = 32, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.question_encoder_shape = question_encoder_shape
        self.answer_encoder_shape = answer_encoder_shape
        self.batch_size = batch_size
        self.question_length = question_length
        self.answer_length = answer_length

        self.model = self.init_model()

    def init_model(self):
        input_question = Input(shape = (None, self.num_features))
        input_ans = Input(shape = (None, self.num_features))

        conv1d_1 = Conv1D(200, 5, activation = "tanh", kernel_regularizer = l2(0.0001))(input_question)
        max_pooling_1 = MaxPooling1D()(conv1d_1)
        question_encoder = Bidirectional(LSTM(self.question_encoder_shape, ))(max_pooling_1)
        
        conv1d_2 = Conv1D(400, 5, activation = "tanh", kernel_regularizer = l2(0.0001))(input_ans)
        max_pooling_2 = MaxPooling1D()(conv1d_2)
        conv1d_3 = Conv1D(600, 5, activation = "tanh", kernel_regularizer = l2(0.0001))(max_pooling_2)
        max_pooling_3 = MaxPooling1D()(conv1d_3)
        answer_encoder = Bidirectional(LSTM(self.answer_encoder_shape))(max_pooling_3)
        
        merge = concatenate([question_encoder, answer_encoder])
        
        dense_1 = Dense(128, activation = "relu")(merge)
        dropout_1 = Dropout(0.25)(dense_1)
        dense_2 = Dense(256, activation = "relu")(dropout_1)
        dropout_2 = Dropout(0.25)(dense_2)
        dense_3 = Dense(1, activation ="sigmoid")(dropout_2)

        model = Model(inputs = [input_question, input_ans], outputs = dense_3)
        metrics = ["accuracy"]
        model.compile(optimizer = Adam(self.learning_rate), loss = "binary_crossentropy", metrics = metrics)

        model.summary()
        return model
    
    def DataGenerator(self, X, y):
        i = 0
        while True:
            X_batch_question = []
            X_batch_answer = []
            y_batch = []
            for j in range(self.batch_size):
                if i == X.shape[0]:
                    i = 0
                question = X[i, 0]
                answer = X[i, 1]
                question_embs, answer_embs = make_w2vec_matrix(question, answer)
                X_batch_question.append(question_embs)
                X_batch_answer.append(answer_embs)
                y_batch.append(y[i])
                i += 1
                #print(np.array(X_batch_question).shape)
            yield [X_batch_question, X_batch_answer], np.array(y_batch)

    def train(self, X, y):
        #prepare data
        X_train, X_val, y_train, y_val = train_test_split(X,y)
        train_generator = self.DataGenerator(X_train, y_train)
        validation_generator = self.DataGenerator(X_val, y_val)

        model_dir = './checkpoints/'
        callbacks = [
            #tf.keras.callbacks.EarlyStopping(patience = 8, monitor = 'val_acc', restore_best_weights = True),
            tf.keras.callbacks.TensorBoard(log_dir = "./logs"),
            tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(model_dir, "weights-epoch{epoch:02d}-loss{val_loss:.2f}-acc{val_acc:.2f}.h5"))
            ]
        history = self.model.fit_generator(generator = train_generator, epochs = 50, verbose = 1, callbacks = callbacks,
            validation_data = validation_generator, steps_per_epoch = X_train.shape[0]//self.batch_size,
            validation_steps = X_val.shape[0]//self.batch_size)
        
        return history
    
    def predict(self, data):
        csv_data = []
        for row in data:
            for paragraph in row['paragraphs']:
                pred = self.model.predict(paragraph['x'])
                if pred >= 0.5:
                    csv_data.append([row['__id__'], paragraph['id']])


if __name__ == '__main__':
    qa = QA_selection(num_features = 400, question_length = 10, answer_length = 50, question_encoder_shape = 20, answer_encoder_shape = 70, 
        batch_size = 1, learning_rate = 0.001)
    #read data
    X, y = read_train_data("./data/train.json")
    #train
    hist = qa.train(X,y)
    qa.model.save("model.h5")

    
