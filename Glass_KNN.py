# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:36:02 2022

@author: Karuna Singh
"""

import pandas as pd # for data manipilation
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for visualizations

# importing the dataset
glass = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\KNN\\glass.csv")

# EDA
glass.head() # checking top 5 records
glass.isna().sum() # Checking missing values
glass.info() # Checking data details
glass.describe() # Checking statistical dimensions of the data
glass.duplicated().sum() # checking for duplicate values


glass = glass.iloc[:,:] # Excluding id column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass.iloc[:, :])
glass_n.describe() # checking data stats after normalization

X = np.array(glass_n.iloc[:,:]) # Predictors 

Y = np.array(glass['Type']) # Target 


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(Y)
encoded

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


# from the plot it is evident that K=7 will give the best model

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred
accuracy_score(Y_test, pred)
