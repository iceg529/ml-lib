# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:02:08 2017

Simple Linear Regression with One Variable

Using Least Square Error Method as Cost function

Check out below youtube video for the derivation 
 of the coefficients O(0) and O(1) :
     https://www.youtube.com/watch?v=Hi5EJnBHFB4
 
 O(0) = mean(Y) - O(1) * mean(X) 
 
 O(1) = sum(Variance of X and Y)/sum(Covariance of X)

Variance of X and Y = sum((x(i)-mean(x))*(y-mean(y)))
                    
@author: Naresh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def variance(x , y):
    # Calculate mean and size as part of the formula requirement
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.sum((x - x_mean) * (y - y_mean))

def plot(x , y, b0, b1):
    plt.scatter(x,y, marker ='o', color ='red')
    y_pred = b0 + b1 * x
    plt.plot(x , y_pred, color = 'blue')
    plt.legend('Linear Regression with One Variable')
    plt.xlabel('Input')
    plt.ylabel('Output')


def get_coeff(x ,y):
    b1 = variance(x,y)/variance(x,x)
    b0 = np.mean(y) - b1 * np.mean(x)
    return (b0 , b1)

def main():
    #Observations
    df = pd.read_csv('listed_actual.csv')
#    x = df.iloc[:,0]
#    y = df.iloc[:,1]

    from sklearn.cross_validation import train_test_split
    df_train,df_test = train_test_split(df,test_size = 0.2)
    
    x_train = df_train.iloc[:,0]
    y_train = df_train.iloc[:,1]
    (b0,b1) = get_coeff(x_train, y_train)
    print("Coefficients are {} and {}".format(b0,b1))
    x_test = df_test.iloc[:,0]
    y_test = df_test.iloc[:,1]
    plot(x_test,y_test,b0,b1)
    y_pred = b0 + b1 * x_test
    
    
if __name__ == "__main__":
    main()