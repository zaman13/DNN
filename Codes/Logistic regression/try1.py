#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:38:53 2025

@author: Mohammad Asif Zaman

"""

import numpy as np
import matplotlib.pyplot as py
import h5py

from LR_dataload import * 


learning_rate = 0.005
num_iter = 7000

# train_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")



train_x_org, train_y_org, test_x_org, test_y_org, classes = load_dataset()
'''
x data structure = [sample_no, pixels, pixels, RGB channel no]
y data structure =  bool output = 1 x sample no.
'''

N_train = train_x_org.shape[0]
Nx = train_x_org.shape[1]
Ny = train_x_org.shape[2]
Nc = train_x_org.shape[3]
N_test = test_x_org.shape[0]

# testing the image
# ind = 38
# py.imshow(train_x_org[ind])
# print(train_y_org[0,ind])

# Flatten the datasets to 1D and normalize
train_x = train_x_org.reshape(Nx*Ny*Nc, N_train)/255
test_x = test_x_org.reshape(Nx*Ny*Nc, N_test)/255
train_y = train_y_org
test_y = test_y_org


# defining the necessary functions
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost(y,yhat):
    M = y.shape[1]
    J = -np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))/M
    return J


def forward(W,B,X,Y):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    J = cost(Y,A)

    dZ = A - Y
    dW = np.dot(X, dZ.T)/X.shape[1]
    dB = np.sum(dZ)/X.shape[1]
    
    return dW, dB, J

def grad_descent(W,B,X,Y, N_iter, learning_rate):
    cost = []
    for iter in range(N_iter):
        iter_percentage = 100*iter/N_iter
        if iter_percentage % 5 ==  0:
            print('Percentage completed =',iter_percentage, '%')
        
        dW, dB, J = forward(W,B,X,Y)
        W = W - learning_rate*dW
        B = B - learning_rate*dB
        cost.append(J)
    
    return W, B, dW, dB, cost

def predict(W,B,X):       
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    return (A >= 0.5)*1.0


def model(X_train, Y_train, X_test, Y_test, num_iter, learning_rate):
    
    # intialization 
    W = np.zeros([train_x.shape[0],1])
    B = 0
    W,B, dW, dB, costs = grad_descent(W, B, train_x, train_y, num_iter, learning_rate)
    
    Y_prediction_train = predict(W,B,X_train)
    Y_prediction_test = predict(W,B,X_test)
    
    train_accuracy = (1-np.mean(np.abs(Y_train - Y_prediction_train)))*100.0
    test_accuracy = (1-np.mean(np.abs(Y_test - Y_prediction_test)))*100.0
    
    print('Train accuracy = ', train_accuracy, '%')
    print('Test accuracy = ', test_accuracy, '%')
    
    py.plot(costs)
    return 0 

d = model(train_x, train_y, test_x, test_y, num_iter, learning_rate)

# # testing
# W, B, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
# dW, dB, J = forward(W,B,X,Y)
# W,B, dW, dB, costs = grad_descent(W, B, X, Y, 100, 0.009)

# Running
# dW, dB, J = forward(W,B,train_x, train_y)
# W,B, dW, dB, costs = grad_descent(W, B, train_x, train_y, max_iter, learning_rate)


    
# py.plot(costs) 
