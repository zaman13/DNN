#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:14:32 2025

@author: Mohammad Asif Zaman

Start date: May 8, 2025

July 12, 2025
 - Added back propagation
 - Added multiple activation function
 - Streamlined arguments of function (removed unnecessary parameter inputs)
 - Added print statements to display status and matrix sizes
"""

import numpy as np

import matplotlib.pyplot as py
from tqdm import tqdm

from LR_dataload import * 
import preprocess_v0_1 as pp

py.close('all')
#==============================================================================
# Loading data
#==============================================================================

print('Loading and pre-processing data..')

# train_x_org, train_y_org, test_x_org, test_y_org, classes = load_dataset()
# train_x, train_y, test_x, test_y = flatten_dataset(train_x_org, train_y_org, test_x_org, test_y_org)


path = '/home/asif/pCloudDrive/ML_Datasets/'
data_folder = 'cat_dog'
N_test = 400
N_train = 1400
reNx = 64
reNy = 64
train_x, train_y, test_x, test_y = pp.pre_process_dataset(path, data_folder, reNx, reNy, N_train, N_test)


print('done\n')

'''
x data structure = [sample_no, pixels, pixels, RGB channel no]
y data structure =  bool output = 1 x sample no.
'''


#==============================================================================




# define neural network parameters
N_layer = 5  # number of layers including input and output layers
N_a = np.zeros(N_layer)   # array that will indicate number of nodes in each layer
# N_a[0] = 3      # input data dimension/ number of featuers
# N_a[1] = 4      # number of nodes/elements in the hidden layer
# N_a[2] = 1      # number of nodes/elements in output layer
N_a = np.array([train_x.shape[0], 128, 32, 4, 1])  # compact notation for nodes in each layer. The first number corresponds to the input layer
# N_a = np.array([train_x.shape[0], 4, 1])  # compact notation for nodes in each layer. The first number corresponds to the input layer
# N_a = np.array([train_x.shape[0], 32, 4, 1])  # compact notation for nodes in each layer. The first number corresponds to the input layer


# N_a = np.array([3, 2, 1])  # compact notation for nodes in each layer

# Activation function in different layers
# 1 = relu, 2 = tanh, 3 = sigmoid, 0 = nothing, only for the first layer. 
# act_fun = [0,1,3]      
act_fun = [0,1,1,2,3]      
# act_fun = [0,1,1,3]      

# Note: act_fun[0] should never be called

# Gradient descent parameters
N_iter = 5000           # number of iterations 
learning_rate = 0.005   # learning rate












#==============================================================================
# Initializer function. Creates the W and B matrix/vectors in the form of lists
#==============================================================================
def initializer(N_a,act_fun):
    # Notes: July 12, 2025: The W and B quantities are independent of the number of samples.
    # They are properties of the NN skeleton itself.
    
    amp_fct = 0.1
    
    W = []  # this will be a list of matrices. Each index will correspond to a layer
    B = []  # this will be a list of vectors. Each index will correspond to a layer
    # A = []  # list of matrices
    
    W.append([])  # W[0] is empty (corresponds to input layer). Starts from W[1]
    B.append([])  # B[0] is empty (corresponds to input layer). Starts from B[1]
    
    # A.append([])  # A[0] is kept empty for now. It should be replaced with the input matrix X later.
    
    N_layer = len(N_a)  # number of layers
    print('Initializing....')
    print(f'No. of layers = {N_layer}')
    print('------------------------------\n') 
    for m in range(1,N_layer):
        # zero value initialization
        # W.append( np.zeros((N_a[m],N_a[m-1])) )
        # B.append( np.zeros((N_a[m],1)) )
        
        
        # random value initialization
        if act_fun[m] == 1:
            temp1 = np.random.randn(N_a[m], N_a[m-1]) * np.sqrt(2 / N_a[m-1])
        else:
            temp1 = np.random.randn(N_a[m], N_a[m-1]) * np.sqrt(1 /  N_a[m-1])

    

        # temp1 = amp_fct * np.random.randn(N_a[m],N_a[m-1])
        W.append( temp1 )
        # temp2 = amp_fct * np.random.randn(N_a[m],1)
        temp2 = np.zeros((N_a[m], 1))
        B.append( temp2 )
        
        
        print(f'Layers = {m}')
        print(f'size(W[{m}]) = {np.shape(W[m])}')
        print(f'size(B[{m}]) = {np.shape(B[m])}')
        print('\n')

        # print('m = %i' %m)
        # print('N_a[m] = %i' %N_a[m])
        # print('N_a[m-1] = %i' %N_a[m-1])
        # print(W[m])
    print('Done')
    print('------------------------------\n') 
    return W, B
#==============================================================================


#==============================================================================
# Activation functions
#==============================================================================

def relu(z):
    out = np.maximum(0,z)     # relu activation function. Note: np.max() and np.maximum() are different. np.max() won't work here
    d_out = np.where(z > 0, 1, 0)    # derivative
    return out, d_out

def tanh(z):
    out = np.tanh(z)   # hyperbolic tan function
    d_out = 1 - out**2 # derivative
    return out, d_out

def sigmoid(z):
    
    # out = 1/(1 + np.exp(-z))  # sigmoid function
    out = np.where(z >= 0, 1 / (1 + np.exp(-z)),  np.exp(z) / (1 + np.exp(z)))  # stable sigmoid for large negative z values
    # try:
    #     out = np.where(z >= 0, 1 / (1 + np.exp(-z)),  np.exp(z) / (1 + np.exp(z)))  # stable sigmoid for large negative z values
    # except RuntimeWarning:
    #     print('Error. Min value of z is:')
    #     print(np.min(z))
            
    d_out = out*(1-out)       # derivative
    return out, d_out


# Activation function 
def activation(z,act_select):
    if act_select == 1:
        return relu(z)
    if act_select == 2:
        return tanh(z)
    if act_select == 3:
        return sigmoid(z)
    if act_select == 0:
        print('Error! act_fun[0] called')
    
    
#==============================================================================


def cost(y,yhat):
    M = y.shape[1]   # number of samples\
    
    epsilon = 1e-15  # Small constant to avoid division by zero
    yhat_c = np.clip(yhat, epsilon, 1 - epsilon)
    # yhat_c = yhat    
        
    J = -np.sum(y*np.log(yhat_c) + (1-y)*np.log(1-yhat_c))/M
    
    # J = (1./M) * (-np.dot(y,np.log(yhat_c).T) - np.dot(1-y, np.log(1-yhat_c).T))

    # J = np.squeeze(J)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    # assert(J.shape == ())
    
    #A[L] = y, dA[L] = dJ/dA[L] = dJ/dy

    # dAL = - y/yhat_c + (1-y)/(1-yhat_c)
    dAL = - (np.divide(y, yhat_c) - np.divide(1 - y, 1 - yhat_c))


    return J, dAL


# Normalizing Z values along a given layer
def batch_norm(Z):
    gamma = 1
    beta = 0
    epsilon = 1e-12
    
    mu = np.mean(Z, axis = 1, keepdims = True)
    var = np.var(Z, axis = 1, keepdims = True);
    Z_norm = (Z-mu)/np.sqrt(var + epsilon)
    
    return gamma*Z_norm + beta

#==============================================================================
# Forward function
#==============================================================================

def forward(Wset,Bset,X):
    A = []    # Initialize list. A[0] will be the input layer, A[-1] will be the output layer. The rest are intermediate layers
    A.append(X) # Assign input layer
    
    Z = []         # Z values for each layer
    Z.append([])   # Append first element of the Z list to be empty
    
    
    # print('Forward propagating....')
    # print('------------------------------\n') 
    
    # print(f'size(Z[0]) = {np.shape(Z[0])}')
    # print(f'size(A[0]) = {np.shape(A[0])}')

    
    N_L = len(Bset)  # number of layers. Note len(Bset) = N_layers
    for m in range(1,N_L):   # loop over layers. 
        Z_raw = np.dot(Wset[m], A[m-1]) + Bset[m]
        # Z_norm = batch_norm(Z_raw)  # call batch normalization
        Z.append(Z_raw)
        # Z.append(np.dot(Wset[m], A[m-1]) + Bset[m])
        temp, d_temp = activation(Z[m], act_fun[m]  )
        A.append(temp)
        # print(f'size(Z[{m}]) = {np.shape(Z[m])}')
        # print(f'size(A[{m}]) = {np.shape(A[m])}')
        
    # print('Done')
    # print('------------------------------\n') 
    
    return A,Z


 

def backward(Wset, Bset, A, Z, dAL):
    
    N_L = len(Bset) # number of layers. Note len(Bset) = N_layers
    M = np.shape(dAL)[1] # number of samples
    
    dA = [0]* N_L   # empty list of N_L elements. We need to define the length of this list first because 
                    # we will assign element values to it from the end
    dZ = [0]* N_L
    dW = [0]* N_L
    dB = [0]* N_L
    
    L = N_L - 1    # number of layers excluding input layer (output layer + hidden layers)
    
    dA[L] = dAL

    
    # print('Backward propagating....')
    # print('------------------------------\n') 
    
   
    
    for m in range(L,0,-1):
        
        g, d_g= activation(Z[m], act_fun[m])

        dZ[m] = dA[m]* d_g  
        dW[m] = (1/M)*np.dot(dZ[m], A[m-1].T)
        dB[m] = (1/M)*np.sum(dZ[m], axis = 1, keepdims = True)
        dA[m-1] = np.dot(Wset[m].T, dZ[m])
        # print(f'size(dZ[{m}]) = {np.shape(dZ[m])}')
        # print(f'size(dA[{m}]) = {np.shape(dA[m])}')
        # print(f'size(dW[{m}]) = {np.shape(dW[m])}')
        # print(f'size(dB[{m}]) = {np.shape(dB[m])}')

    # print(f'size(dA[{0}]) = {np.shape(dA[0])}')    

    # print('Done')
    # print('------------------------------\n')   
        
    return dA, dZ, dW, dB


def grad_descent(W,B,X,Y, N_iter, learning_rate):
    print('Gradient descent')
    clip_value = 10
    N_L = len(B)   # number of layers
    cost_store = []
    for iter in tqdm(range(N_iter)):
        
        iter_percentage = 100*iter/N_iter
        # if iter_percentage % 5 ==  0:
            # print('Percentage completed =',iter_percentage, '%')
        
        # Forward propagation
        A,Z = forward(W,B,X)
        
        # Cost evaluation
        J, dAL = cost(Y,A[-1])  # Y_hat is the last A, hence A[-1]
        cost_store.append(J)
        
        # Backward propagation
        dA, dZ, dW, dB = backward(W, B,A, Z, dAL)
        
        # Loop over the layers
        for m in range(1,N_L):
            dW[m] = np.clip(dW[m], -clip_value, clip_value)
            dB[m] = np.clip(dB[m], -clip_value, clip_value)
            
            W[m] = W[m] - learning_rate*dW[m]
            B[m] = B[m] - learning_rate*dB[m]
        
        # if iter % 100 == 0:
        #     print(f"Iteration {iter}: Cost = {J}")
        #     print(f"Max dW: {max([np.max(np.abs(dw)) for dw in dW[1:]])}")
        #     print(f"Max dB: {max([np.max(np.abs(db)) for db in dB[1:]])}")

    return W, B, dW, dB, cost_store, A[-1]

def predict(W,B,X):
    A,Z = forward(W,B,X)
    return (A[-1] >= 0.5)*1.0
    

def model(N_a, act_fun, X_train, Y_train, X_test, Y_test, num_iter, learning_rate):
    
    # intialization 
    W, B = initializer(N_a, act_fun)
    
    W,B, dW, dB, costs, AL = grad_descent(W, B, X_train, Y_train, num_iter, learning_rate)
    
    Y_prediction_train = predict(W,B,X_train)
    Y_prediction_test = predict(W,B,X_test)
    
    train_accuracy = (1-np.mean(np.abs(Y_train - Y_prediction_train)))*100.0
    test_accuracy = (1-np.mean(np.abs(Y_test - Y_prediction_test)))*100.0
    
    # print('------------------------------\n') 
    print('Train accuracy = ', train_accuracy, '%')
    print('Test accuracy = ', test_accuracy, '%')
    
    
    return W, B, AL, costs

W, B, AL, cost_store = model(N_a, act_fun, train_x, train_y, test_x, test_y, N_iter, learning_rate)

# Plotting cost function
py.plot(cost_store)
py.xlabel('iterations')
py.ylabel('cost value')


# plotting how training data matches with prediction. also plot the activation (pre thresholding)
aa, zz = forward(W, B, train_x)
pp = predict(W,B,train_x)
py.figure()
py.plot(aa[-1][0],'b', label = 'AL')
py.plot(train_y[0],'rx', label = 'train_y')
py.plot(pp[0], 'ko', label = 'predict')
py.title('train')

# plotting how test data matches with prediction. also plot the activation (pre thresholding)
aa, zz = forward(W, B, test_x)
pp = predict(W,B,test_x)
py.figure()
py.plot(aa[-1][0],'b', label = 'AL')
py.plot(test_y[0],'rx', label = 'test_y')
py.plot(pp[0], 'ko', label = 'predict')
py.title('test')



# W1_img = W[1].T.reshape(W[1].shape[0], 64, 64, 3)

# W, B = initializer(2, [2,1])
# W[1], B[1], X, Y = np.array([[1.],[2.]]).T, 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

# J = forward(W,B,X,Y)
# print(J)







# # Test case
# X = np.array([[1.0],
#               [0.5],
#               [-1.5]])
# Y = np.array([[1.0]])

# W[1] = np.array([[0.2, -0.4, 0.1],
#                [0.5, 0.3, -0.2],
#                [-0.3, 0.8, 0.7],
#                [0.6, -0.1, 0.2]])  # shape (4, 3)

# B[1] = np.array([[0.1],
#                [0.2],
#                [0.0],
#                [-0.1]])            # shape (4, 1)

# W[2] = np.array([[0.4, -0.6, 0.3, 0.2]])  # shape (1, 4)
# B[2] = np.array([[0.05]])                # shape (1, 1)


# print(W)
# print(B)





# W, B = initializer(N_a)
# # M = 10
# # X = np.random.randn(N_a[0],M)
# # Y = np.random.randn(N_a[-1],M)
# M = N_train
# X = train_x
# Y = train_y 

# # A,Z = forward(W,B,X)
# # J, dAL = cost(Y,A[-1])  # Y_hat is the last A, hence A[-1]
# # dA, dZ, dW, dB = backward(W, B,A, Z, dAL)

# W,B, dW, dB, costs = grad_descent(W, B, X,Y, N_iter, learning_rate)

# py.plot(costs)




