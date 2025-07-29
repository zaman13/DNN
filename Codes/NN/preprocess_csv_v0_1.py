#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:42:23 2025

@author: Mohammad Asif Zaman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_csv(path, fname, split_ratio, th):

    rnd_state = 12
    
    # read files    
    df = pd.read_csv(path + fname)
    
    # separate features and target columns
    X = df.drop('quality', axis=1).values  # replace 'target_column'
    y = df['quality'].values  

    
    y = y > th
    
    # py.plot( ymod[0,:])
    
    
    # Normalize the features so that they have zero mean and unit variance
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    
    
    # Split training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=rnd_state)
    
    
    # reshape the variables in the correct format for downstream NN processing
    
    X_train = X_train.T  # reshape into (N,m) instead of (m,N)
    X_test = X_test.T    # reshape into (N,m) instead of (m,N)
    y_train = y_train.reshape(1,-1) # reshape into (1,m) instead of (m,)
    y_test = y_test.reshape(1,-1)  # reshape into (1,m) instead of (m,)
 

    return X_train, y_train, X_test, y_test


# path = 'datasets/'
# fname = 'WineQT.csv'

# split_ratio = 0.3
# th = 5.5


# X_train, y_train, X_test, y_test = preprocess_csv(path, fname, split_ratio, th)

