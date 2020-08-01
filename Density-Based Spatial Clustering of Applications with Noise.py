#!/usr/bin/env python
# coding: utf-8

# # Implementing DBSCAN Clustering

# ##### Perform DBSCAN clustering from vector array or distance matrix.
# 
# DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.

# In[1]:


#Importing required libraries


import matplotlib.pyplot as plt   
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.datasets import make_moons


# In[2]:


data, labels = make_moons(n_samples=200,noise=0.15)


# In[3]:


#plot the dataset to see how it actually looks


plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=labels) 


# In[4]:


from sklearn.cluster import DBSCAN


# In[5]:


clustering = DBSCAN(eps=0.3,min_samples=10).fit(data)
labels = clustering.labels_
print(labels)


# In[6]:


#plot the clusters to see how actually our data has been clustered


plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=clustering.labels_, cmap='rainbow') 


# In[7]:


new_data = np.array([data[i] for i in range(len(labels)) if labels[i]!=-1])
new_labels = [labels[i] for i in range(len(labels)) if labels[i]!=-1]


# In[8]:


new_data.shape


# In[9]:


plt.figure(figsize=(10, 7))  
plt.scatter(new_data[:,0], new_data[:,1], c=new_labels, cmap='rainbow') 


# In[10]:


## MANASH PRATIM KAKATI
## E&ICT ACADAMY, IIT GUWAHATI
## PG CERTIFICATION IN AI & ML


# In[ ]:




