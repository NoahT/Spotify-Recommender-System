#!/usr/bin/env python
# coding: utf-8

# In[140]:


import numpy as np
import pandas as pd
# import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
pd.options.display.max_rows = 10000


# In[2]:


# In[3]:


df = pd.read_csv('../dataset/data_o.csv')
df
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


# In[142]:


df[-10000:]


# In[201]:


'''
Bring components together into retrieval algorithm
@param data pandas datafram with song names
@param data without song names
@param principal_coordinates numpy matrix, where row i denotes the
principal coordinates used for row i in data
@param k number of results to retrieve
@param song user would like to find most similar song to
@returns song name of most similar song
'''
def retrieve(data, data_clean, principal_coordinates, k, name):

    # Error check if song does not exist in database
    if (data['name']== name).any() == False:
        print("Sorry, that song does not exist in our database")

    df=data_clean
    results = principal_coordinates.T

    # Index for song we are looking at
    r_num = data.index[data['name']== name].tolist()[0]
    #dimension used w/ PCA
    d_size = results.shape[1]

    # Perform KNN on principal coordinates - get the closest coordinates
    neighbors = get_nn(results, r_num, k)
    neighbors = neighbors[:k]
    retrieval_name = [data['name'][neighbor[1]] for neighbor in neighbors]
    retrieval_artists = [data['artists'][neighbor[1]] for neighbor in neighbors]
    return (retrieval_name, retrieval_artists)


# In[1]:


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
    


# In[208]:


'''
kNN algorithm, which takes in a dataset and a specified
row and returns the k nearest neighbors.
@param dataset Numpy matrix, where the rows denote entries
@param target_row_index Row index, denoting the row in dataset that we will
base our nearest neighbors results off of
@param k integer, denoting the number of neighbors to return
@returns tuply array, where the first entry denotes the distance
from our target row and the second entry denotes the row index of
the corresponding nearest neighbors
'''
def get_nn(dataset, target_row_index, k):
    target_row = dataset[target_row_index,:]
    ''' 
    Return k nearest neighbors
    '''
    #total number of rows
    row_count = dataset.shape[0]
    
    #empty vector of Euclidean distances
    distances = []
    
    #Populate empty vector
    for i in range (0,row_count-1): 
        if(i != target_row_index):
            dist = e_dist(target_row, dataset[i])
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
    


# In[209]:


def e_dist(r1,r2):
    ''' 
    Return Euclidian distance with parameters r1,r2 (row 1 and 2)
    '''
    row_len = len(r1)
    r1 = r1.reshape((row_len,1))
    r2 = r2.reshape((row_len,1))
    
    distance  = 0.0
    for i in range(row_len):
        #sq_dist += (r1.iat[0,i] - r2.iat[0,i])**2 
        distance += (r2[i] - r1[i])**2  
    return (np.sqrt(distance))  


# In[8]:


'''
Function to compute the principal coordinates and
principal components of the given dataset.
@param data numpy matrix, where each column denotes a single entry
@param k integer value denoting the dimension with which we
should perform dimension reduction
@returns numpy matrix, where the ith column denotes the principal coordinates
of the ith row in data
'''
def principal_component_analysis(data, k):
    # number of entries N in our dataset
    N = data.shape[0]
    mean = np.mean(data, axis=1).reshape((N,1))
    data_centralized = data - mean
    # compute covariance matrix 
    cov_mat = (data_centralized @ data_centralized.T)/N
    # compute principal components
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # change the order of eigen_values in descending order (dominant pairs first)
    eigen_values = np.flip(eigen_values, axis=0)
    # get the k dominant eigenvectors (principal components)
    principal_components = eigen_vectors[-k:]
    # gets the principal components in descending order
    principal_components = np.flip(principal_components, axis=1)
    # matrix of principal components
    V = np.column_stack(principal_components)
    # get principal coordinates of data
    principal_coords = V.T.dot(data_centralized)

    return (principal_coords, eigen_values)


# In[68]:


'''
Function to compute the scree plot
of the eigenvalues computed from PCA. We use
this for exploratory data analysis in order to
infer the number of principal coordinates that
should be used for dimensionality reduction.
@param eigen_vals numpy array containing the
eigenvalues of the principal coordinates calculated
from PCA
@param export_name export name for scree plot figure
'''
def scree(eigen_vals, export_name):
    num_vals = eigen_vals.shape[0]
    sum_eigen_vals = np.sum(eigen_vals)
    proportion = np.array([eigen_val / sum_eigen_vals for eigen_val in eigen_vals])
    x_ticks = np.arange(1,num_vals+1)
    plt.plot(x_ticks, proportion)
    plt.title("Scree plot of eigen values for principal components")
    plt.savefig(export_name)


# In[113]:


'''
Catalog coverage method used in order to
determine the proportion of items in our dataset
that are recommended over time. Catalog coverage
is calculated by performing N rankings of length k,
and computing the cardinality of the union of the
elements recommended across all recommendations.
@param N integer denoting the number of recommendations to
perform
@param k integer denoting the size of our rankings for
each recommendation
@param dataset Numpy matrix, where the rows denote entries
'''
def catalog_coverage(N, k, dataset):
    coverage = []
    # number of entries in our dataset
    list_size = dataset.shape[0]
    items_recommended = np.zeros(list_size)
    for i in range(N):
        # sample a random entry in our dataset
        row_index = np.random.randint(0,list_size)
        recommendations = get_nn(dataset, row_index, k)
        indices = [recommendation[1] for recommendation in recommendations]
        for recommendation in recommendations:
            recommendation_index = recommendation[1]
            items_recommended[recommendation_index] = 1
        current_coverage = np.sum(items_recommended) / list_size
        print('Coverage for iteration', i+1, ':\t', current_coverage, )
        coverage.append(current_coverage)
    return np.array(coverage)


# In[1]:


N=1000


# In[131]:


pca_coverage_rank1 = catalog_coverage(N, 1, vecs.T)


# In[132]:


pca_coverage_rank5 = catalog_coverage(N, 5, vecs.T)


# In[133]:


pca_coverage_rank10 = catalog_coverage(N, 10, vecs.T)


# In[134]:


x_ticks = np.arange(pca_coverage_rank1.shape[0])
plt.title('Catalog coverage for rankings of size k')
plt.xlabel('Recommendations observed')
plt.ylabel('Percentage of dataset observed')
plt.plot(x_ticks, pca_coverage_rank1 * 100, label='k=1')
plt.plot(x_ticks, pca_coverage_rank5 * 100, label='k=5')
plt.plot(x_ticks, pca_coverage_rank10 * 100, label='k=10')
plt.legend(loc=0)
plt.savefig('pca catalog coverage')
plt.show()


# In[69]:


(vecs, vals) = principal_component_analysis(df_clean.to_numpy().T, 1)


# In[215]:


names, artists = retrieve(df, df_clean, vecs, 10, 'Land Of Canaan')


# In[216]:


rankings_df = pd.DataFrame(data={'Song':np.array(names).T, 'Artists': artists})
rankings_df.index.name = 'Rank'
rankings_df


# In[ ]:





# In[85]:


rankings = retrieve(df, df_clean, vecs, 10, 'Harvest Moon')


# In[86]:


rankings


# In[ ]:





# In[12]:


plt.scatter(vecs[0,:],vecs[1,:], s=.1)
plt.title("Leading two principal coordinates for points in dataset")
plt.savefig('is_this_correct')


# In[13]:


scree(vals, 'scree_plot')


# In[ ]:




