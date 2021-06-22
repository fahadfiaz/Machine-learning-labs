#!/usr/bin/env python
# coding: utf-8

# #### System specifications on which experiments were run!!! Ram-> 4GB, Core i5

# #### I was unable to process all the 128 datasets because it was taking alot of time to process some datasets, So I just run this task on 12 datasets in repository!!!. As the code works for first 12 datasets, it will also work for other datasets.

# ## Importing Libraries

# In[59]:


import pandas as pd
import numpy as np
from sklearn import neighbors
import math
import warnings
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter
import functools
plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore")


# ### Load all datasets in repository and get different statistics about each dataset. Also get list of datasets having missing value or unequal length.

# In[60]:


path = "./UCRArchive_2018"
dir_list = os.listdir(path) 
DatasetNames = []

dir_list.remove("Missing_value_and_variable_length_datasets_adjusted")
for datasets_name in dir_list:
    
    DatasetNames.append(datasets_name)


# ## Datasets that we are going to use for this exercise!!!

# In[61]:


sampled_datasets = DatasetNames[:12]
sampled_datasets[2] = "DiatomSizeReduction"
sampled_datasets[3] = "Coffee"
sampled_datasets[4] = "Computers"
sampled_datasets


# ### Function to split the dataset into train, test and validation sets

# In[62]:


def split(df,targetcol):
    
    X = df
    Y = targetcol 
    X_copy = X.copy()
    
    Xtrain = X_copy.sample(frac=0.70, random_state=0)
    Xtest = X_copy.drop(Xtrain.index)
    Xtest_copy = Xtest.copy()
    Xtest = Xtest_copy.sample(frac=0.50, random_state=1)
    Xvalidate = Xtest_copy.drop(Xtest.index)
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

# In[63]:


def eucl_dist2(a , b):
    distance = np.sum(np.square(a-b),axis=1)
    return np.sqrt(distance)


# In[64]:


def eucl_dist(a , b):
    distance = np.sum(np.square(a-b))
    return np.sqrt(distance)


# ### Function that will impute the missing values with the the mean of its nearest K neighbors

# In[65]:


def replace_missing_values_Knn(df_train, k):
    X = df_train.copy()
    X = X.dropna(axis='columns')

    for i, row in X.iterrows():
        
        if df_train.iloc[i,:].isnull().values.any():        
            unique_id = i
            other_rows = X[~X.index.isin([i])]
            res = eucl_dist2(X, row)
            srt = np.argsort(res)[1:k+1]
            replacesment = np.mean(res[srt])
            df_train.iloc[i,:].fillna(replacesment, inplace=True)

    return df_train


# ### Function to calculate the k- nearest distances of query with training dataset with specific distance measure.

# In[66]:


def best_nn(df , query , k, distName):
    distance =[]
    if k > df.shape[0]:
        k = df.shape[0]
    
    ts = pd.DataFrame({0:query}).T
    
### Pdist function only calculate pairwise distances between each observations of input matrix. To make it work in our case
### we have to concatenate query row with row of train dataset, pass this stacked matric to Pdist function and do this operation
### for each row of train dataset.This approach was very slow and was taking alot of time. 

### So to make this approach faster, I stacked the query row on top of train dataset matrix and then found the distance using
### following approach which was quite faster then processing individual row of train matrix with query row.
    

    df_concat = pd.concat([ts, df], axis=0) 
    
### pdist will give us Pairwise distances between each observations.Then we use squareform function to convert this pairwise
### distance vector to a square-form distance matrix. Now each row is a pairwise distance between each observation. 
### The distance between query row and train matrix will be found at 0 index of distance matrix. We ignore the 0 index because it
### is a distance of query row from query row which will be zero. So we ignore the 0 index of final distance list

    distance = squareform(pdist(df_concat,distName))[0][1:]

    agg = np.argsort(distance)[:k]
    return agg


# ### Function to  calculate k-neareast neighbours and do majority voting to get final prediction

# In[67]:


def prediction(x , y , query, k, distName):
    predictions = []
    
    for index,row1 in query.iterrows():        
        k_min = best_nn(x , row1 , k, distName)
        m = y[k_min].tolist()
        predictions.append(max(m , key=m.count))
    return predictions


# ### Function to get accuracy on specific dataset

# In[68]:


def get_accuracy(a , p):

    count = 0
    for i ,j in zip(a , p):
        if (i == j):
            count += 1
    return (count/a.shape[0]) * 100


# ### All distance measure provided by scipy library

# In[69]:


distNames = ["","euclidean", "minkowski", "cityblock", "seuclidean", "sqeuclidean","cosine","correlation","hamming","jaccard",
            "chebyshev","canberra","braycurtis","yule","matching","dice","kulsinski","rogerstanimoto","russellrao",
             "sokalmichener", "sokalsneath"]


# ### Main Function

# In[70]:


try:
    for ds in sampled_datasets:
    
        train_pt = "./UCRArchive_2018\\" + ds + "\\"+ ds + "_TRAIN.tsv"
        test_pt = "./UCRArchive_2018\\" + ds + "\\"+ ds + "_TEST.tsv"
        df_train = pd.read_csv(train_pt, sep='\t', header=None)
        df_test = pd.read_csv(test_pt, sep='\t', header=None)
        df_full = df_train.append(df_test)
        df_full.reset_index(drop=True, inplace=True)
        df_train = df_full.iloc[:,1:]
        df_test = df_full.iloc[:,0]

        print("##############################################--",ds," Dataset -- Started\n")
        print("Full Dataset Shape ",df_full.shape )
        print( "Check if dataset has null values or missing values!!! ",np.isnan(df_train.values).any()) 
        print("\nK-nearest neighbour started!!!")

        listAccr = np.zeros((11,21)) # Outer list index is to get optimal k-for prediction, Inner list will save results of predictions 
                                     # using different distance measure!!!

        for k1 in range(0,10): # grid search to get best k for doing prediction
            for k2 in range(0,20): # use different distance measures for prediction

                Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = split(df_train,df_test) # Split the dataset!!!

                pred = prediction(Xtrain, np.array(Ytrain)[0], Xval, k1+1 , distNames[k2+1]) # Do prediction with each distance 
                                                                                             #  measure.

                listAccr[k1+1,k2+1] = get_accuracy(np.array(Yval)[0], pred)

            print("Predictions done with k: ", k1+1)

        print("K-nearest neighbour finished!!!")
        print("Final validation accuracies after grid search '(x-axis index - k used for prediction, y-xis index- k used for imputing!!!)'\n")
        print(listAccr[1:,1:])

        indx = np.unravel_index(np.nanargmax(listAccr), listAccr.shape) # get maximum accuracy index

        best_accr = [[i,np.argmax(sublist),max(reversed(sublist))] for i, sublist in enumerate(listAccr)]
        best_metric = sorted(best_accr, key=itemgetter(2), reverse=True)
        k_best , distance_type, accuracy = best_metric[0]

        print("\n Optimal K for prediction: ",k_best, ", Optimal distance measure for prediction: ", distNames[distance_type], ", Accuracy on these optimal paramters: ",accuracy )

        print("\nCalculating test accuracy by using optimal k value for prediction along with optimal distance measure!!!")

        ### Now do prediction on test set with optimal paramters!!!

        Xtrain0, Ytrain0, Xval0, Yval0, Xtest0, Ytest0 = split(df_train,df_test)

        pred = prediction(Xtrain0, np.array(Ytrain0)[0], Xtest0, k_best, distNames[distance_type] )

        print("\nFinal accuracy on test data: ", get_accuracy(np.array(Ytest0)[0], pred))
        print("\nFinished --",ds," Dataset--##############################################\n\n")
except:
    pass
    


# ### Assumption:
# 

# <p>For each dataset above, I have shown validation accuracies on all the distance measures provided by scipy library. For each k value, I calculated the best distance measure for that specific K-VALUE. After that, I calculated the overall best distance measure on all k values. Then I have shown the testing accuracy by using optimal K value and best distance measure. In exercise question, it is mentioned that we have to show the result of all distances on test set also. But, this thing does not make any sense. If we have to show the results of all distance measures on test set also, then what is the reason of finding the best distance measure from training set!!!.</p>

# In[ ]:




