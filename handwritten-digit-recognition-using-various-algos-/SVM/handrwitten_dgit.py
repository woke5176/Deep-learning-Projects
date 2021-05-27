#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:10:45 2020

@author: adarsh
"""

import sys
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve, KFold , cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

#print(os.listdir("./input"))

old_stdout = sys.stdout
log_file = open("summary.log", "w")
sys.stdout = log_file


test_data = pd.read_csv("./input/test.csv")
train_data = pd.read_csv("./input/train.csv")

print("Shapes")
print(train_data.shape)
print(test_data.shape)
print("head of train")

print(train_data.head(10))

print("check for null entries")

print(train_data.isnull().sum().head(10))

print(train_data.isnull().sum().head(10))
print("description of data")
print(train_data.describe())

print("about datatype -train")
print("Dimension: ", train_data.shape, "\n")
print("about datatype -test")
print("Dimensions: ", test_data.shape, "\n")


print("labels")

print(list(np.sort(train_data['label'].unique())))

print("plots")

sns_plot = sns.countplot(train_data['label'])
sns_plot.figure.savefig("label.png")

print("plotting an instance of dataset \n\n")

four  = train_data.iloc[3, 1:]
#print("four shape" , four.shape ,"\n")
four = four.values.reshape(28,28)
four_image = plt.imshow(four, cmap='gray')
four_image.figure.savefig("four.png")
plt.title("digit 4")


print(round(train_data.drop('label', axis=1).mean(),2))


print("separting x and y")

y = train_data['label']
X = train_data.drop(columns = 'label')

print("X shape", X.shape, "\n")
print("y shape", y.shape, "\n")

print(y.head())


print("Normaliozation")

X = X/255.0
test_data = test_data/255.0


print("Scaling features")
from sklearn.preprocessing import scale
X_scaled = scale(X)

print("\ntrain-test split")
X_train, X_test , y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, train_size = 0.2, random_state = 10)

print("building models")
'''
print("\nlinear")
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

print("\nprediction")
y_pred = model_linear.predict(X_test)
print(y_pred)

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
print(metrics.confusion_matrix(y_true=y_test, y_pred = y_pred))
'''

print("non - linear")
print("\n using rbf kernel, without hyperparameter tuning")

non_linear_model = SVC(kernel = 'rbf')
non_linear_model.fit(X_train, y_train)

y_pred = non_linear_model.predict(X_test)
print("predictions")
print(y_pred)

print("performance")

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))


print("Hyperparameter tuning (C and gamma)")

folds = KFold(n_splits=5, shuffle=True, random_state=10)

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],'C': [5,10]}]

model = SVC(kernel='rbf')

model_cv = GridSearchCV(estimator=model, param_grid=hyper_params, scoring='accuracy', cv = folds, verbose=1, return_train_score=True)

print("Fitting the model") 
model_cv.fit(X_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)

print(cv_results)

        


