# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:34:44 2019

@author: Savitoj
"""
import time
import pandas as pd

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics as sm


def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))

def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    print("svm started")
    svc.fit(X_train,y_train)
    print(str(time.time() - start)/60)
    y_pred=svc.predict(X_test)
    calc_accuracy("SVM",y_test,y_pred)

def predict_lr(X_train, X_test, y_train, y_test):
    clf = lr()
    print("lr started")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    calc_accuracy("Logistic regression",y_test,y_pred)


def predict_nb(X_train, X_test, y_train, y_test):
    clf = nb()
    print("nb started")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    calc_accuracy("Naive Bayes",y_test,y_pred)


def predict_knn(X_train, X_test, y_train, y_test):
    clf=knn(n_neighbors=8)
    print("knn started")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    calc_accuracy("K nearest neighbours",y_test,y_pred)

def predict_mlp(X_train, X_test, y_train, y_test):
    clf=mlp()
    print("mlp started")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    calc_accuracy("MLP classifier",y_test,y_pred)

df = pd.read_csv("asl_dataset_orb.csv", sep=',',header=None)
X = df.iloc[:, :-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)


start = time.time()

predict_svm(X_train, X_test,y_train, y_test)
'''
predict_knn(X_train, X_test,y_train, y_test)
predict_lr(X_train, X_test,y_train, y_test)
predict_nb(X_train, X_test,y_train, y_test)
predict_mlp(X_train, X_test,y_train, y_test)
'''



