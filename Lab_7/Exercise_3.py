#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import warnings
warnings.filterwarnings("ignore")
import os
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter
import functools
import time


# ## Utility Functions

# ### Function to split the dataset into train, test and validation sets

# In[2]:


def split(df,targetcol):
    
    X = df
    Y = targetcol 
    X_copy = X.copy()
    
    Xtrain = X_copy.sample(frac=0.70, random_state=0)
    Xtest = X_copy.drop(Xtrain.index)
    Xtest_copy = Xtest.copy()
    Xtest = Xtest_copy.sample(frac=0.50, random_state=1)
    Xvalidate = Xtest_copy.drop(Xtest.index)
    
    # Normalizing features 
    norm_Xtrain = np.linalg.norm(Xtrain, axis = 1, keepdims = True)
    Xtrain = Xtrain / norm_Xtrain
    norm_Xtest = np.linalg.norm(Xtest, axis = 1, keepdims = True)
    Xtest = Xtest / norm_Xtest
    norm_Xval = np.linalg.norm(Xvalidate, axis = 1, keepdims = True)
    Xval = Xvalidate / norm_Xval
    
    Y_copy = Y.copy()
    Ytrain = Y_copy.sample(frac=0.70, random_state=0)
    Ytest = Y_copy.drop(Ytrain.index)
    Ytest_copy = Ytest.copy()
    Ytest = Ytest_copy.sample(frac=0.50, random_state=1)
    
    Yvalidate = Ytest_copy.drop(Ytest.index)
    
    Ytrain = np.matrix(Ytrain)
    Ytest = np.matrix(Ytest)
    Yval = np.matrix(Yvalidate)
    
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


# ### Function to calculate the euclidean distance

# In[3]:


def eucl_dist(a , b):
    distance = np.sum(np.square(a-b))
    return np.sqrt(distance)


# In[4]:


def eucl_dist2(a , b):
    distance = np.sum(np.square(a-b),axis=1)
    return np.sqrt(distance)


# ### Function to calculate the distance measure point wise

# In[5]:


### This function take three paramters -> one row of training dataset, one row of validation dataset and maximum distance 
###                                    -> till that time.

### First it calculate the the euclidean distance between respective columns of two rows. Then it check if this distance is 
### greater than maximum distance till that time, it will break loop and return the flag false along with maximum distance. This
### is why this method runs faster than normal k-nearest neighbours method.

### But, if that is not the case, then it will save the distance and do this procedure for all columns. If the loop does not
### break, then it means the distance between the current two rows is minimum then this function return true flag along with 
### the minimum calculated distance

def dist_measure(a , b, max_dist_till_now):
    total_dist = 0
    
    for i,j in zip(a,b):
        difference = np.sqrt(np.square(i-j))
        total_dist += difference
        if total_dist > max_dist_till_now:
            break;
    
    if total_dist > max_dist_till_now:
        return False, max_dist_till_now
    else:
        return True,total_dist


# ### Function to calculate the k- nearest distances

# <p>This function calculate the k neareast distances. It first calculate the distance of query row with k- rows of train dataset and save it in distance array tp. Then for each other row of train dataset, it call dist_measure method.If dist_measure method returns true then it replace the maximum distance from k-nearest distances array with distance return by dist_measure method.</p>

# In[6]:


def best_nn(df , query , k):
    tp = np.zeros(df.shape[0])

    specific_rows = df.iloc[:k,:]
    i=0
    for index,row in specific_rows.iterrows():
        tp[i] = eucl_dist(row, query)
        i=i+1
    
    df_c = df.copy()
    remaining_rows = df_c.drop(specific_rows.index)
    
    
    for index,row in remaining_rows.iterrows():
        
        flag, dist =  dist_measure(row, query, max(tp))
        if flag:
            ind = np.argmax(tp)
            tp[ind] = 0
            tp[i] = dist
            
        i = i + 1
    
    agg =  np.nonzero(tp)[0]
    return agg


# ### Function to  calculate k-neareast neighbours and do majority voting to get final prediction

# In[7]:


def prediction(x , y , query, k):
    predictions = []
    
    for index,row1 in query.iterrows():
        k_min = best_nn(x , row1 , k)
        m = y[k_min].tolist()
        predictions.append(max(m , key=m.count))
    return predictions


# ### Function to get accuracy on specific dataset

# In[8]:


def accuracy(a , p):

    count = 0
    for i ,j in zip(a , p):
        if (i == j):
            count += 1
    return (count/a.shape[0]) * 100


# In[ ]:





# ### Loading Crop Dataset

# #### Crop dataset was the largest dataset in the repository. We can confirm it in exercise 0 

# In[9]:


train_pt = "./UCRArchive_2018\\" + "Crop" + "\\"+ "Crop" + "_TRAIN.tsv"
test_pt = "./UCRArchive_2018\\" + "Crop" + "\\"+ "Crop" + "_TEST.tsv"
df_train = pd.read_csv(train_pt, sep='\t', header=None)
df_test = pd.read_csv(test_pt, sep='\t', header=None)
df_full = df_train.append(df_test)
df_full.reset_index(drop=True, inplace=True)
print("Full Dataset Shape ",df_full.shape )
print( "Check if dataset has null values or missing values!!! ",np.isnan(df_train.values).any()) 


# #### Full Dataset was very large and took alot of time to perform experiments so I have downscaled the dataset by doing stratified sampling. I took 20 samples per class.

# In[10]:


df2 = df_full.groupby(df_full.columns[0]).apply(lambda x: x.sample(20))
for i in range (1,25):
    print("Total sample belonging to class:" , i , "are ",df2[df2[0]==i].shape[0])


# In[11]:


print("Full dataset after sampling: ",df2.shape )


# In[12]:


df2.reset_index(drop=True, inplace=True)


# ### Seperate train and prediction dataset

# In[13]:


df_train = df2.iloc[:,1:]
df_test = df2.iloc[:,0]
print("Full Training Dataset Shape ",df_train.shape)
print("Full Testing Dataset Shape ",df_test.shape)
print( "Check if dataset has null values or missing values!!! ",np.isnan(df_train.values).any()) 


# ### Run Partial Distances/Lower Bounding K-nearest neighbour algorithm

# In[36]:


start_time = time.time()

listAccr = np.zeros(11)

for k1 in range(0,10):
        
        # Function to split the train, prediction sets to train,test,validation sets
        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = split(df_train,df_test)
        
        pred = prediction(Xtrain, np.array(Ytrain)[0], Xval, k1+1)
        
        listAccr[k1+1] = accuracy(np.array(Yval)[0], pred)
        
        print("processed for k ", k1+1)
print("--- %s seconds ---" % (time.time() - start_time))


# ### Validation set accuracies for each K values. Here list index represent K value!!!

# In[37]:


listAccr


# ### Best K value with highest accuracy

# In[38]:


indx = np.unravel_index(np.nanargmax(listAccr), listAccr.shape)
print("Best accuracy on validation set: ",listAccr[indx[0]], " with k: ",  indx[0])


# ### Testing the best k value on test set

# In[39]:


Xtrain0, Ytrain0, Xval0, Yval0, Xtest0, Ytest0 = split(df_train,df_test)    
pred = prediction(Xtrain0, np.array(Ytrain0)[0], Xtest0, 1)

print('Accuracy on test set', accuracy(np.array(Ytest0)[0], pred))


# ## Now we will run the simple Knn algorithm without lower bounding on crop dataset.

# ### Only these two functions needed to be changed for simple K nearest neighbour

# In[26]:


### Now this function calculate all the distances of query row with traininig dataset and select the min k distances
def best_nn_2(df , query , k):
    distance =[]    

    for j,row2 in df.iterrows():
        distance.append(eucl_dist(row2, query))        
    agg = np.argsort(distance)[:k]
    return agg


# In[27]:


def prediction2(x , y , query, k):
    predictions = []
    
    for index,row1 in query.iterrows():        
        k_min = best_nn_2(x , row1 , k)
        m = y[k_min].tolist()
        predictions.append(max(m , key=m.count))
    return predictions


# ### Run the simple knn algorithm

# In[28]:


start_time = time.time()
listAccr = np.zeros(11)
for k1 in range(0,10):

        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = split(df_train,df_test)
        
        pred = prediction2(Xtrain, np.array(Ytrain)[0], Xval, k1+1)
        
        listAccr[k1+1] = accuracy(np.array(Yval)[0], pred)
        
        print("processed for k ", k1+1)
print("--- %s seconds ---" % (time.time() - start_time))


# ### Validation set accuracies for each K values. Here list index represent K value!!!

# In[29]:


listAccr


# ### Best K value with highest accuracy

# In[30]:


indx = np.unravel_index(np.nanargmax(listAccr), listAccr.shape)
print("Best accuracy on validation set: ",listAccr[indx[0]], " with k: ",  indx[0])


# ### Testing the best k value on test set

# In[32]:


Xtrain0, Ytrain0, Xval0, Yval0, Xtest0, Ytest0 = split(df_train,df_test)
    
pred = prediction2(Xtrain0, np.array(Ytrain0)[0], Xtest0, 4)
print('Test Accuracy ', accuracy(np.array(Ytest0)[0], pred))


# In[ ]:





# ## Result Analysis!!!

# #### When we use partial distances/lower bounding k nearest neighbour algorithm, we got total algorithm running time of 43.7 seconds which is less than simple k nearest neighbour algorithm which had 135.9 seconds total running time. These results showed that  partial distance/lower bounding knn is infact faster than simple knn. But it comes at the cost of accuracy as accuracy of simple knn was better than partial distance knn. 
# #### But I have run the experiments on the downscale dataset so, its possible that when we run the experiments on full dataset, then running time difference may by significantly larger among two algorithms and accuracy difference may be smaller.

# In[ ]:




