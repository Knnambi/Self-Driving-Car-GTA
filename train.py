# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:54:44 2018

@author: Admin
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import cv2
import os
import pickle
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
import random
import h5py
   
with open('my_datasetimg.pickle', 'rb') as data:
    dataset = pickle.load(data)
X=np.array(dataset)

with open('my_dataset.pickle', 'rb') as datakey:
    datasetkey = pickle.load(datakey)
Y=np.array(datasetkey)

#X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
# create model
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160, 320, 3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1,activation='relu'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
# Fit the model
model.fit(X, Y, epochs=100, batch_size=10)
model.save('model_15.h5')
# evaluate the model
scores = model.evaluate(X, Y)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)
# joblib.dump(clf, 'model.pkl')

# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))