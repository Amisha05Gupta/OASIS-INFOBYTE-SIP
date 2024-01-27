#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


#dataset
iris=pd.read_csv("Iris.csv")
iris.head()


# In[3]:


iris.describe()


# In[4]:


#getting unique values from SPECIES
iris["Species"].unique()


# In[5]:


#splitting data into training and testing
X = iris.drop('Species', axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


X_train


# In[7]:


X_test


# In[8]:


y_train


# In[9]:


y_test


# In[10]:


#training using KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)


# In[11]:


#predicting test 
y_pred = knn_classifier.predict(X_test)


# In[12]:


y_pred


# In[13]:


accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:",accuracy)

