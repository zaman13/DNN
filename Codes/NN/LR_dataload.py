#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:40:21 2025

@author: asif
"""

import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def flatten_dataset(train_x_org, train_y_org, test_x_org, test_y_org):
    
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
    # Note: Flattening setup has a huge impact! How the different pixels are organized matters a lot!
    
    # # Flatten 1: only about 34% on test data set
    # train_x = train_x_org.reshape(Nx*Ny*Nc, N_train)/255
    # test_x = test_x_org.reshape(Nx*Ny*Nc, N_test)/255
    
    # Flatten 3: 70% on test database (July 13, 2025)
    train_x = train_x_org.reshape(train_x_org.shape[0],-1).T/255
    test_x = test_x_org.reshape(test_x_org.shape[0],-1).T/255
    
    
    train_y = train_y_org
    test_y = test_y_org
    
    return train_x, train_y, test_x, test_y
    #==============================================================================