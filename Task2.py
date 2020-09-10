# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:33:40 2020

The Sparks Foundation

Task # 2 I had to predict the marks scored when the number of hours studied by 
the student has been provided.is a machine learning problem (unsupervised 
learning). It implements simple linear regression


@author: Walter Stevens

"""
# Importing the necessary libraries

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
path = Path('Users/Angela/documents/GRIP')
from sklearn.metrics import r2_score
from sklearn import metrics


# Importing the dataset
dataset = pd.read_csv(Path.cwd()/'student_scores - student_scores.csv')

# Inspecting the dataset:
dataset.shape
dataset.describe()
    
    
#Splitting the data into the dependent and independent variables
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression


regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Scores vs Hours (Training set)')
viz_train.xlabel('Hours of Study')
viz_train.ylabel('Scores')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Scores vs Hours (Test set)')
viz_test.xlabel('Hours of Study')
viz_test.ylabel('Scores')
viz_test.show()

#To retrieve the intercept:
#print(regressor.intercept_)
#2.5069547569547623
#For retrieving the slope:
#print(regressor.coef_)
#[9.69062469]

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred)),2)
#Mean Absolute Error: 4.691397441397438
print('Mean Squared Error:', round((metrics.mean_squared_error(y_test, y_pred))),2)
#Mean Squared Error: 25.463280738222547
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
#Root Mean Squared Error: 5.046115410711743
print('R Squared:',round(r2_score(y_test, y_pred),2))

#Finally, the test case:
example = 9.25
f = float(regressor.intercept_ + (example * regressor.coef_))

print('The estimated score for a student who spent',example,'hours studying, is',round(f,2))

