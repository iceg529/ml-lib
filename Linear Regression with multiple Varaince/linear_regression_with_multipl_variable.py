# Linear Regression with Multiple Variable
"""
Multi-variant Linear Regression 

Since we have multiple Variables, we do Feature Scaling
    Rescaling - Scale based on Min/Max
    Mean Normalisation - Scale with Min/Max and mean

Derivation of coefficients is available in:
    Stardford Coursera ML course
    https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf

@author: Naresh
"""
import numpy as np
import pandas as pd

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
    coeff = get_constant_coeff(X,y)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X,y)
    y_prefect_pred = regressor.predict(X)
    y_pred = np.sum(np.multiply(np.transpose(coeff) , X),axis = 1)
    print(y_pred)
    type(np.array(y_pred))
    np.shape(y.values[:,0] )
    np.shape(np.array(y_pred))
    np.corrcoef(y.values[:,0] , np.array(y_pred))
    np.corrcoef(np.array(y_prefect_pred[:,0]), np.array(y_pred))
    
    
if __name__ == "__main__":
    main()