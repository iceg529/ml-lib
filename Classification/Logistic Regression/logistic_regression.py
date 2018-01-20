# Logistic Regression for Classification
"""
Logistic Regression

Got this data set accuracy was 89.64%

@author: Naresh
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

def scale(X):
    min_X = np.min(X)
    max_X = np.max(X)
    mean_X = np.mean(X)
    return (X-mean_X)/(max_X-min_X)


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def add_X0(X):
    n,m = np.shape(X)
    X0 = np.ones((n,1))
    return np.hstack((X0,X))

def main():
    dataset = load_breast_cancer()
    X = dataset['data']
    Y = dataset['target'].reshape(569,1)
    X = add_X0(X)
    learning_rate=0.01
    params = np.random.rand(31,1)*0.01
    for i in range(0,10000):
        y_pred = sigmoid(np.dot(X,params))
        dParams=np.dot(X.T,(y_pred-Y))
        params=params-learning_rate*dParams
    y_pred = sigmoid(np.dot(X,params))
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y, y_pred)
    
if __name__ == "__main__":
    main()