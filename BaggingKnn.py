#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:08:25 2019

@author: sai
"""
#import numpy as np

from sklearn.ensemble import BaggingClassifier #import the classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier #import KNN Classifier


KNN_bag = KNeighborsClassifier()

bag_knn = BaggingClassifier(KNN_bag) #set n_estimators=10

#-------------------------------------------------------------#

#bag_knn=BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=10, max_samples=0.5, bootstrap=True,random_state=3) 

from sklearn.datasets import load_breast_cancer #importing datsets from sklearn


dataset=load_breast_cancer() #load breast cancer dataset

X=dataset.data #assign X 
#np.isnan(X).sum() Finding Null Values

y=dataset.target #assign y to target
#np.isnan(y).sum() Finding the Null Values

from sklearn.model_selection import train_test_split #for splitting the datset
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=3) #split the dataset

#----------------------KNeighborsClassifier()---------------------------------------#
parameters = {'n_neighbors' : [5, 10, 50, 100, 200, 500],
              'weight' : ['uniform', 'distance'],
              'algorithm' : ['balltree', 'kd_tree', 'brute', 'auto'],
              'p' : [1, 2],
              'n_jobs' : [1, 2, -1],
              }

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = KNN_bag,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)


#-----------------------BaggingClassifier()-----------------------------------------#
parameters = {'n_estimators' : [10, 50, 100, 200, 500],
              'max_features' : [1.0, 2.0, 3.0, 5.0],
              'bootstrap_features' : ['False', 'True'],
              'bootstrap' : ['True', 'False'],
              'n_jobs' : [1, 2, -1],
              'random_state' : [0, 1, 10, 100]
              }


grid_search = GridSearchCV(estimator = bag_knn,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracies = grid_search.best_score_


'''
y_pred = bag_knn.predict(X_test)


bag_knn.fit(X_train, y_train) #fit method to fit the dataset

bag_knn.score(X_test, y_test) #check the score on test set

knn=KNeighborsClassifier(n_neighbors=5) #let's try to see the result on a single k-NN classifier on the data

knn.fit(X_train, y_train) #fit

knn.score(X_test, y_test) #score on test data
'''


