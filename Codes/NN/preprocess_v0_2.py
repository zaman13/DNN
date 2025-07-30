#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:48:05 2025

@author: Mohammad Asif Zaman

Deep learning preprocessing pipeline

# Input Arguments
    path = path of the dataset folder
    data_folder = name of the dataset folder
    reNx, reNy = resized dimensions of the images.Example: 64 x 64
    N_train = Number of training samples 
    N_test = Number of test samples
    
- July 29, 2025
  - combining image preprocessing and csv data preprocessing into a single file

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as py


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

    


def preprocess_img(path, data_folder, reNx, reNy, N_train, N_test):
    
    
    
    split_ratio = N_train/(N_test + N_train)

    
    # 1. Define transforms: resize, convert to tensor, flatten
    transform = transforms.Compose([
        transforms.Resize((reNy, reNx)),              # Resize all images to 64x64
        transforms.ToTensor(),                    # Convert to tensor (C x H x W)
        # transforms.Lambda(lambda x: x.view(-1))   # Flatten to 1D
        transforms.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1)) # Reorder to (H x W x C), then flatten
    ])
    
    # 2. Load dataset from directory
    dataset = datasets.ImageFolder(root=path + data_folder, transform=transform)
    
    # Splitting dataset into training and testing subsets
    
    torch.manual_seed(13)   # setting a fixed seed so that we get the same split everytime
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Dataloaders
    
    # N_train = len(train_dataset) // 2
    # N_test = len(test_dataset) // 2
    
    train_loader = DataLoader(train_dataset, batch_size=N_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=N_test, shuffle=False)
    
    images, labels = next(iter(train_loader))  
    X_train = images.numpy().T
    Y_train = labels.numpy().reshape(1, -1)  # reshape to (1,m) vector instead of (m,) vector for later compatability
    
    images, labels = next(iter(test_loader))  
    X_test = images.numpy().T
    Y_test = labels.numpy().reshape(1, -1) # reshape to (1,m) vector instead of (m,) vector for later compatability
    

    return X_train, Y_train, X_test, Y_test


#Define dataset location directory and folder
# path = '/home/asif/pCloudDrive/ML_Datasets/'
# data_folder = 'cat_dog'
# N_test = 200
# N_train = 400
# reNx = 64
# reNy = 64

# x1, y1, x2, y2 = pre_process_dataset(path, data_folder, reNx, reNy, N_train, N_test)
# rnd_ind = 31
# img = x1[:,rnd_ind]
# # img = img.reshape(3,reNy, reNx)
# # img = img.transpose(1,2,0)
# img = img.reshape(reNy, reNx,3)

# py.imshow(img)

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


