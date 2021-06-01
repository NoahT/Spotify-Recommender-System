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

# %%
#remove all categorical data
df_clean = df.drop(['artists','id','key','mode','name', 'release_date', 'explicit'],axis=1)

#convert release_date to just year
#YYYY-MM-DD
#YYYY
def first_four(year):
    out_year = ""
    for i in range(0,4): 
        out_year += year[i]
    return out_year
        
    
#df['release_date'] = df['release_date'].apply(first_four)
df_clean['tempo'] = df_clean['tempo'].apply(np.round)


# In[4]:


df_clean


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

#computes top eigenpairs
def power_method(A, num_pairs, iterations):
    
    #initializaing lists to store eigenpairs
    top_eigenvec = []
    
    top_eigenval = []
    
    #initial nonzero approximation
    x = np.random.rand(A.shape[0],1)
    
    #iterating based on how many dominant components we want to find 
    for j in range(num_pairs):
        
        #how many iterations we want to use to converge to the best approximation
        for i in range(iterations):
        
            x = np.dot(A, x)
        
            eigenvec_norm = np.linalg.norm(x)
            eigenvec = x / eigenvec_norm
        
            eigenval = (np.dot(A,x).T.dot(x)) / np.dot(x.T,x)
        
        top_eigenvec.append(eigenvec)
        top_eigenval.append(eigenval)
        
        #deflation is used to compute further eigenpairs
        A = A - eigenval * ((np.dot(eigenvec, eigenvec.T)) / (np.dot(eigenvec, eigenvec.T)))
   
    
    return top_eigenvec, top_eigenval
    


# %%
#df=df.T
# %%
#results = principal_component_analysis(df, 10)

# %%
#results = results.T
# %%
#Euclidean distance 
def e_dist(r1,r2):
    ''' 
    Return Euclidian distance with parameters r1,r2 (row 1 and 2)
    '''
    row_len = len(r1)
    r1 = r1.reshape((row_len,1))
    r2 = r2.reshape((row_len,1))
    
    distance  = 0.0
    for i in range(row_len-1):       
        #sq_dist += (r1.iat[0,i] - r2.iat[0,i])**2 
        distance += (r2[i] - r1[i])**2  
    return (np.sqrt(distance))  
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
        dist = e_dist(test_row, dataset[i])
        distances.append((dataset[i],dist, i))
        
    #sort by placing smallest distances at top/beginning of vector    
    distances.sort(key=lambda tup: tup[1])
    #print(distances)
    
    #Empty Vector or neighbors
    neighbors = []
    #Populate with nearest neigbors (neighbors with smallest distance) within range k 
    for i in range(k): 
        neighbors.append((distances[i][0], distances[i][2]))
    return neighbors
        
# %%
#Run kNN

#row number
#r_num = 100
#number of neighbors
#k = 1
#dataset size
#d_size = len(results[0])

#k+1 becasue k[0] returns r_num for some reason. Will fix later
#neighbor = get_nn(results, results[r_num:r_num+1].reshape((d_size, 1)), k+1)

#Starting at index 1 for same reason as above 
#neighbor
# %%
#e_dist(results[100].reshape((10,1)), results[0].reshape((10,1)))
# %%
#distance = 0.0
#for i in range(len(results[100].reshape((10,1)))-1):
#	distance += (results[100].reshape((10,1))[i] - results[0].reshape((10,1))[i])**2
# %%
#distance
# %%
#np.sqrt(distance)
# %%
#e_dist(results[100], results[0])
# %%
'''
Bring components together into retrieval algorithm
@param data pandas datafram with song names
@param data without song names
@param song user would like to find most similar song to
@returns song name of most similar song
'''
def retrieve(data, data_clean, name):

    # Error check if song does not exist in database
    if (data['name']== name).any() == False:
        print("Sorry, that song does not exist in our database")

    # perform principal component analysis
    df=data_clean.T
    # Optimal value? Just chose 5
    results = principal_component_analysis(df, 5)
    results = results.T

    # Index for song we are looking at
    r_num = data.index[data['name']== name].tolist()[0]
    # Number of neighbors
    k = 1
    #dataset size
    d_size = len(results[0])

    # Perform KNN on principal coordinates - get the closest coordinates
    # k+1 becasue k[0] returns r_num for some reason. Will fix later
    neighbor = get_nn(results, results[r_num:r_num+1].reshape((d_size, 1)), k+1)
    neighbor = neighbor[1]

    # Retrieve the song name based off of the index in the neighbors list
    return data['name'][neighbor[1]]

# %%
''' 
Example for one of the songs in the dataset
1st run: received "Tell Me More-More-Then Some
'''
retrieve(df, df_clean, 'Dolorosa')

