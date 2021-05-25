#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# import scipy.io as sio
# import matplotlib
# import matplotlib.pyplot as plt
# from numpy.matlib import repmat
# from sklearn.preprocessing import normalize


# In[3]:


df = pd.read_csv('../dataset/data_o.csv')
df
#remove all categorical data
df = df.drop(['artists','id','key','mode','name'],axis=1)

#convert release_data to jusrt year
def first_four(year):
    out_year = ""
    for i in range(0,4): 
        out_year += year[i]
    return out_year
        
    
df['release_date'] = df['release_date'].apply(first_four)
df['tempo'] = df['tempo'].apply(np.round)


# In[4]:


df


# In[ ]:




