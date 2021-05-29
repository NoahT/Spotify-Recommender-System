#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
# from numpy.matlib import repmat
# from sklearn.preprocessing import normalize


# In[3]:


df = pd.read_csv('../dataset/data_o.csv')
df
#remove all categorical data
df = df.drop(['artists','id','key','mode','name'],axis=1)

#convert release_date to just year
#YYYY-MM-DD
#YYYY
def first_four(year):
    out_year = ""
    for i in range(0,4): 
        out_year += year[i]
    return out_year
        
    
df['release_date'] = df['release_date'].apply(first_four)
df['tempo'] = df['tempo'].apply(np.round)


# In[4]:


df


# In[5]:

'''
Function to compute the principal coordinates and
principal components of the given dataset.
@param data numpy matrix, where each row denotes a single entry
@param k integer value denoting the dimension with which we
should perform dimension reduction
@returns numpy matrix, where the ith row denotes the principal coordinates
of the ith row in data
'''
def principal_component_analysis(data, k):
    data_centralized = data - np.mean(data, axis=0)
    # number of entries N in our dataset
    N = data.shape[0]
    print(N)
    # compute covariance matrix 
    cov_mat = (data_centralized @ data_centralized.T)/N
    print(cov_mat.shape)
    # compute principal components
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # get the k dominant eigenvectors (principal components)
    principal_components = eigen_vectors[-k:]
    # gets the principal components in descending order
    principal_components = np.flip(principal_components, axis=1)
    # matrix of principal components
    V = np.column_stack(principal_components)
    # get principal coordinates of data
    principal_coords = V.T.dot(data_centralized)

    return principal_coords
