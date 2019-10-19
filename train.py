from clean_data import read_train_data, read_test_data
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Flatten, GlobalAveragePooling1D
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np

def init_model():
    model = Sequential()
    model.add(Conv1D(100, 10 ,activation='relu'))
    # model.add(Conv1D(100, 10, activation='relu'))
    model.add(GlobalMaxPooling1D())
    # model.add(Conv1D(160, 10, activation='relu'))
    # model.add(Conv1D(160, 10, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

def evaluate(model, X,y):
    count = 0
    for i in range(len(X)):
        pred = model.predict(X[i])
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == y[i]:
            count += 1
    return (count * 1.0) / len(y)


def main():
    #Read train data
    X, y = read_train_data('train.json')
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.01)
    # X_train, X_val, y_train, y_val = X[:len(X)-1000], X[len(X)-1000:], y[:len(y) - 1000], y[len(y) - 1000:]
    #Init model
    model = init_model()

    # Train
    # for epoch in range(10):
    #     print("Epoch %d: " % epoch)
    #     for i in range(len(X_train)):
    #         print(X_train[i].shape)
    #         model.fit(np.array([X_train[i]]), np.array(y_train[i]), verbose = 0)
    #     print('Train Acc: %.3f%' % evaluate(model, X_train, y_train))
    #     print('Val Acc: %.3f%' % evaluate(model, X_test, y_test))

    # Train
    print(X_train.shape)
    model.fit(X_train, y_train, validation_split=0.1, batch_size=128, epochs = 50, verbose = 0)

    #Predict on test data and write to csv
    csv_data = []
    tests = read_test_data('test.json')
    for test in tests:
        for paragraph in test['paragraphs']:
            pred = model.predict(paragraph['x'])
            if pred >= 0.5:
                csv_data.append([test['__id__'], paragraph['id']])
    df = pd.DataFrame(csv_data, columns = ['test_id', 'answer']) 
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()