# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:04:19 2020

@author: Walter Stevens

Python prgram to implement unsupervised learning.Namely, to predict the optimal 
number of custers and represent it visually. The dataset is 

For this exercise I implemented K Means clustering using .sklearn

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from numpy import random, float


# Importing the dataset
dataset = pd.read_csv(Path.cwd()/'Iris.csv')

diction = {"Iris-setosa":0,"Iris-versicolor":1,'Iris-virginica':2}
#since the dataset contained the actual sub-species names e.g. Iris-setosa,
#I had to replace those categorical variables with  numeric ones (9,1,2)

dataset.replace({"Species": diction},inplace = True)

#Exploratory data analysis
dataset.shape
dataset.describe()
#plotting the data as is
s = dataset['Species']
y = s.values.tolist()
X = dataset

ydf = pd.DataFrame(y,columns = ['Species'])

X.drop('Species', 1, inplace = True)
X.drop('Id', 1, inplace = True)

#I dropped the2 extraneous columns from the X dataset: Species because it
#is the classifier and is non-numeric, and Id because it didn't contain
#relevant data


#Fitting the model
model = KMeans(n_clusters=3).fit(X)
#centroids = model.cluster_centers_
#print(centroids)

#Looking at the clusters each data point was assigned to
#print(model.labels_)  

# Visualising the True Classification and  Model Classification groupings:
colors = np.array(['red', 'green', 'blue'])
predictedY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
plt.subplot(1, 2, 1)
plt.scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=colors[ydf['Species']], s=40)
plt.title('True classification')

plt.subplot(1, 2, 2)
plt.scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=colors[predictedY], s=40)
plt.title("Model's classification")







