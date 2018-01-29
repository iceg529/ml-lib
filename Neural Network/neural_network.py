# The starting point for Neural Network
"""
Created on Sat Jan 27 09:44:32 2018

The input must be having n features and m training samples

@author: Naresh
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

def main():
    dataset = load_breast_cancer()
    X = dataset.data
    m = X.shape[0]
    n = X.shape[1]
    Y= dataset.target
    Y = Y.reshape(m,1)
    # Get the size of the neural network
    W1,b1 = initialise_weights(30,30)
    W2,b2 = initialise_weights(30,1)
    learning_rate=0.01
    J=[]
    for i in range(0,1000):
        W1_temp,Z1_temp,A1_temp,W2_temp,Z2_temp,A2_temp = forward_propagation(X,Y,W1,b1,W2,b2)
        current_J=compute_cost(A2_temp,Y)
        if len(J) > 0 and current_J > J[-1]:
            break
        J.append(current_J)
        W1,Z1,A1,W2,Z2,A2 =W1_temp,Z1_temp,A1_temp,W2_temp,Z2_temp,A2_temp
        dZ1,dW1,db1,dZ2,dW2,db2 = backward_propagation(X,A1,A2,Y,W1,W2,Z1)
        Z1=Z1-learning_rate*dZ1
        W1=W1-learning_rate*dW1
        b1=b1-learning_rate*db1
        Z2=Z2-learning_rate*dZ2
        W2=W2-learning_rate*dW2
        b2=b2-learning_rate*db2
    plt.plot(J)
    plt.show()
    _,_,_,_,_,Y_pred = forward_propagation(X,Y,W1,b1,W2,b2)
    from sklearn.metrics import confusion_matrix
    Y_pred = Y_pred.reshape(m,1)
    Y_pred = np.where(Y_pred > 0.5 , True , False)
    cm = confusion_matrix(Y_pred,Y)

def initialise_weights(input_size,output_size):
    # m , n = 5,7
    W = np.random.rand(output_size,input_size)
    b = np.zeros(1)
    return W,b

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def derivative_of_sigmoid(Z):
    computed_value=1/(1+np.exp(-Z))
    return computed_value * (1-computed_value)

def forward_propagation(X,Y,W1,b1,W2,b2):
    Z1 =np.dot(W1,X.T)+b1
    A1 = sigmoid(Z1)
    Z2 =np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    return W1,Z1,A1,W2,Z2,A2

def backward_propagation(X,A1,A2,Y,W1,W2,Z1):
    m = Y.shape[0]
    n = X.shape[1]
    dZ2 = A2-Y.T
    dW2 = np.sum(np.dot(dZ2,A1.T),axis=0)/m
    db2 = dZ2
    dZ1 = np.multiply(np.dot(W2.T,dZ2),derivative_of_sigmoid(Z1))
    dW1 = np.sum(np.dot(dZ1,X),axis=0)/m
    db1=dZ1
    return dZ1,dW1,db1,dZ2,dW2,db2

def compute_cost(AL,Y):
    m =Y.shape[0] 
    return -np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m

if __name__ == "__main__":
    main()