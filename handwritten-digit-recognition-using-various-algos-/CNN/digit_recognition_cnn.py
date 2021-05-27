#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: adarsh
"""

from numpy import mean
from numpy import std
from matplotlib  import pyplot
from sklearn.model_selection import  KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

#function to load training and test data

def load_dataset():
    #load
    (trainX, trainY), (testX, testY) = mnist.load_data()
    #reshape 28 * 28
    trainX = trainX.reshape((trainX.shape[0], 28, 28 , 1))
    testX = testX.reshape((testX.shape[0], 28, 28 , 1))
    #one-hot encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX,testY

#scale pixel values

def prepare_pixel(train,test):
    #int to float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #mormalize
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    
    return train_norm,test_norm

#define model 

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu' , kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    #compiling model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#evaluation of model

def evaluate_model(dataX, dataY, n_folds = 5):
    scores, histories = list(), list()
    #prepare cross_validation
    kfolds = KFold(n_folds, shuffle=True,random_state=1)
    #enumaerate splits
    for train_index, test_index in kfolds.split(dataX):
        #define model
        model = baseline_model()
        #select rows for test and train
        trainX, trainY, testX, testY = dataX[train_index], dataY[train_index], dataX[test_index], dataY[test_index]
        #fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %0.f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories

#plot diagnostic curves

def diagnostic_curves(histories):
    for i in range(len(histories)):
        #plotting loss
        pyplot.subplot(2,1,1)
        pyplot.title('Cross entropy loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()
    
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()
    #save the plot    
	pyplot.savefig('performance1.png')

#running the whole model
def run_model():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prepare_pixel(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	diagnostic_curves(histories)
	# summarize estimated performance
	summarize_performance(scores)
    
run_model()


    
        