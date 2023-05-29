#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random as rd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
from scipy.spatial import distance

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output


# In[ ]:


#loading data

data = pd.read_csv("Data for Problem 2/seeds.txt", sep=" ", header=None)
data.dropna(axis=1,inplace=True)
data.columns = data.columns.map(str)
data.columns=["feature1","feature2","feature3","feature4","feature5","feature6","feature7"]


# In[ ]:


data.head()


# In[ ]:


def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


# In[ ]:


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    distances2=distances.min(axis=1)
    return distances.idxmin(axis=1),sum(distances2*distances2)


# In[ ]:


def new_centroids(data, labels, k):
    centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids


# In[ ]:


def plot_clusters(data, labels, centroids, iteration,k):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    #clear_output(wait=True)
   # plt.title(f'Iteration {iteration}')
    plt.title(f'number of clusters {k}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1],color='Red')
    plt.show()


# In[ ]:


max_iterations = 100

avg_sse_k = {}
for k in [3,5,7]:
    sse_array=[]
    sse_last_element=[]
    j=0
    #condition=True
    for i in range(10): #10 times random inintilazation of centroids
       # print(i, end=", ")
        condition=True
        centroids = random_centroids(data, k)
        old_centroids = pd.DataFrame()
        iteration = 1


        while iteration <= max_iterations and condition : #100 iterations and sse difference <0.001
            old_centroids = centroids
            labels,sse= get_labels(data, centroids)
            sse_array.append(sse)
            if j>0 and iteration !=1: #to avoid comparing the sse of one initialization to the last iteration of last initilisation
              
                if sse_array[j-1]-sse_array[j]<=0.001:
                    condition = False
            j+=1
            centroids = new_centroids(data, labels, k)
            iteration += 1
        sse_last_element.append(sse_array[len(sse_array)-1])
    plot_clusters(data, labels, centroids, iteration,k)
    avg_sse_k.update({k: np.mean(sse_last_element)})


# In[ ]:


avg_sse_k

