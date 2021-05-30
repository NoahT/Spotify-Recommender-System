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



# %%
df["release_date"] = pd.to_numeric(df["release_date"])
# %%
# %%
results = principal_component_analysis(df[0:1000], 10)
# %%
#Euclidean distance 
def e_dist(r1,r2):
    ''' 
    Return Euclidian distance with parameters r1,r2 (row 1 and 2)
    '''
    row_len = r1.shape[1]
    sq_dist  = 0.0
    for i in range(0,(row_len-1)):       
        sq_dist += (r1.iat[0,i] - r2.iat[0,i])**2   
    return (np.sqrt(sq_dist))  
def get_nn(dataset, test_row, k):
    ''' 
    Return k nearest neighbors
    '''
    #total number of rows
    row_count = dataset.shape[0]
    
    #empty vector of Euclidean distances
    distances = []
    
    #Populate empty vector
    for i in range (0,row_count-1): 
        dist = e_dist(test_row, dataset[i:i+1])
        distances.append((dataset[i:i+1],dist))
        
    #sort by placing smallest distances at top/beginning of vector    
    distances.sort(key=lambda tup: tup[1])
    
    #Empty Vector or neighbors
    neighbors = []
    #Populate with nearest neigbors (neighbors with smallest distance) within range k 
    for i in range(k): 
        neighbors.append(distances[i][0])
    return neighbors
        
# %%
#Run kNN

#row number
r_num = 5
#number of neighbors
k = 1

#k+1 becasue k[0] returns r_num for some reason. Will fix later
#Also use subset of data to test
neighbor = get_nn(results, results[r_num:r_num+1].reshape(14,1), k+1)

#Starting at index 1 for same reason as above 
neighbor[1:]
# %%
