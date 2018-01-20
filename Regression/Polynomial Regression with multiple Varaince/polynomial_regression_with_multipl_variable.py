# Linear Regression with Multiple Variable
"""
Multi-variant Polyomial Regression 

Same as Linear Regression with multiple Variable.
But, Adding inputs with Polynomial Features

@author: Naresh
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def scale(X):
    min_X = np.min(X)
    max_X = np.max(X)
    mean_X = np.mean(X)
    return (X-mean_X)/(max_X-min_X)


def get_constant_coeff(X,y):
    X_trans =np.transpose(X)
    return np.dot(np.linalg.pinv(np.dot(X_trans, X)) , np.dot(X_trans , y))

def add_X0(X):
    n,m = np.shape(X)
    X0 = np.ones((n,1))
    return np.hstack((X0,X))

def main():
    dataset = pd.read_csv('movies.csv')
    X = dataset.iloc[:,0:3]
    y = dataset.iloc[:,3:4]
    X = add_X0(X)
    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)
    coeff = get_constant_coeff(X,y)
    y_pred = np.sum(np.multiply(np.transpose(coeff) , X),axis = 1)
    print(y_pred)
    type(np.array(y_pred))
    np.shape(y.values[:,0] )
    np.shape(np.array(y_pred))
    val = np.corrcoef(y.values[:,0] , np.array(y_pred))
    print(y.values[:,0])
    
    
if __name__ == "__main__":
    main()