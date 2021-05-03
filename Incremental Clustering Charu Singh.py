#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale,normalize, StandardScaler
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


iris = datasets.load_iris()
dir(iris)


# In[3]:


iris.data


# In[4]:


iris.data.shape


# In[5]:


iris.feature_names


# In[6]:


X=iris.data
y = pd.DataFrame(iris.target)
clustering = KMeans(n_clusters=3, random_state=10)
clustering.fit(X)


# In[7]:


clustering.labels_


# In[9]:


target_predicted = np.choose(clustering.labels_,[0,1,2]).astype(np.int64)
target_predicted


# In[10]:


from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(iris.target, target_predicted))
confusion_matrix(iris.target, target_predicted)


# In[11]:



iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
y.columns =['Targets']


# In[12]:



iris.target


# In[13]:


color_theme = np.array(['red','blue','green'])

plt.scatter(x=iris_df.petal_length, y=iris_df.petal_width, c= color_theme[iris.target],s=50)
plt.title ("This is Actual Flower Cluster")


# In[14]:



plt.scatter(x=iris_df.petal_length, y=iris_df.petal_width, c= color_theme[target_predicted],s=50)
plt.title ("This is KMeans Clustering ")


# In[15]:



datasets.load_iris().target


# In[16]:



clustering.cluster_centers_


# In[17]:



clustering.labels_


# In[18]:


target_predicted = np.choose(clustering.labels_,[2,0,1]).astype(np.int64)
target_predicted


# In[19]:


confusion_matrix(iris.target,target_predicted)


# In[20]:


from sklearn.metrics import accuracy_score
accuracy_score(iris.target,target_predicted)


# In[21]:


x = [2,3,4,5,6]
y = [8,5,6,3,4]
even = [0,1,0,1,0]
col = np.array(['red','green'])

plt.scatter(x,y, c=col[even])


# In[ ]:




