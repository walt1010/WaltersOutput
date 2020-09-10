# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:04:19 2020

@author: Walter Stevens

An annotated example of XG-boost on the Iris dataset, just to
illustrate the use XG-boost as opposed to  ordinary decision trees. 
XG-Boost uses regularised boosting, can handle missing values automatically, and 
supports parallel processing.

Although I did this by way of comparison with the Spark Foundations unsupervised
learning task (Task 3), this is obviously not unsupervised.

"""
import pandas as pd
from pathlib import Path

# Importing the dataset
dataset = pd.read_csv(Path.cwd()/'Iris.csv')

#Initial Data discovery
no_Samples, no_Features = dataset.shape
print(no_Samples)
print(no_Features)
print(dataset.Species.unique())

diction = {"Iris-setosa":0,"Iris-versicolor":1,'Iris-virginica':2}
#since the dataset contained the actual sub-species names e.g. Iris-setosa,
#I had to replace those categorical variables with  numeric ones (9,1,2)

dataset.replace({"Species": diction},inplace = True)

s = dataset['Species']
y = s.values.tolist()
X = dataset

ydf = pd.DataFrame(y,columns = ['Species'])

X.drop('Species', 1, inplace = True)
X.drop('Id', 1, inplace = True)

#Splitting off 20% for the test data, leaving me with 80%
#for the training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, ydf,test_size=0.2, random_state=0)

#loading up XG-Boost and converting the both groups of data into the DMatrix form that it supports

import xgboost as xgb

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

# Defining the hyperparameters by defining the dictionary. Using softmax since this is a multiple
# classification problem. The other parameters should ideally be tuned
# through experimentation, much like the k count in k means

param = {
'max_depth': 4,
'eta': 0.3,
'objective': 'multi:softmax',
'num_class': 3}
epochs = 10

#NB its not softmax that is minimized in XG-boost, but the crossentropy loss
#function, which is based on softmax. Crossentropy is calculated on a
#softmax output, that's why they are a standard couple in ML.
#Tree-based classifiers like XGB find "cuts", or portions of the variables'
#space in a way that minimizes the entropy of a dataset.

#Training the model
model = xgb.train(param, train, epochs)

#Using the trained model for the predictions
predictions = model.predict(test)

#print(predictions)
#Measuring the accuracy on the test data...
from sklearn.metrics import accuracy_score
Accuracy_Result = accuracy_score(y_test, predictions)
print("The accuracy of the XGBoost model was",Accuracy_Result)
#Returned result of 1 which means perfect accuracy