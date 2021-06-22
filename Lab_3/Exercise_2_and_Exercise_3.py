#!/usr/bin/env python
# coding: utf-8

# ## Exercise 2 

# ### Importing libraries

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path


# ### Part A.) Data Preprocessing

# In[11]:


DatasetBasePath = 'Dataset'


# #### Read Airline and demand dataset

# In[12]:


path_to_airline_dataset =  os.path.join(DatasetBasePath,'airq402.data') 
airline_data=pd.read_csv(path_to_airline_dataset,header = None, delimiter=r"\s+")
airline_data.columns= ['city1', 'city2', 'avg_Fare', 'distance', 'avg_weekly_passengers', 'market_leading', 'market_share', 'avg_fare', 'low_priceair','market share','price']


# #### Read Winequality dataset

# In[10]:



path_to_wine_dataset =  os.path.join(DatasetBasePath,'winequality-red.csv') 
winequality_red_data=pd.read_csv(path_to_wine_dataset, sep=';')


# #### Read Parkinsons dataset

# In[14]:


path_to_wine_dataset =  os.path.join(DatasetBasePath,'parkinsons_updrs.data') 

parkinsons_upd=pd.read_csv(path_to_wine_dataset, sep=',')


# <hr style="border:2px solid gray"> </hr>

# In[23]:


airline_data.dtypes


# <p>All three datasets are loaded as required. Only the airfare and demand dataset had non-numeric
# columns <b> (city1, city2, market_leading, low_priceair)</b> that were converted into numeric columns using ‘one-hot encoding’.</p>
# <p>Build-in pandas function get_dumies  was used to change categorical data present in the dataset to one hot encoding. E.g, if you have a column with four categories data then it will replace that column with four new columns each representing each category as 1 while other values will be zero.</p>

# ### Converting Non Numeric values in Airline Data to Numeric Using One-Hot Encoding

# In[24]:


#dummy_values=pd.get_dummies(airline_data)
cols_to_transform = [ 'city1', 'city2', 'market_leading', 'low_priceair']
df_with_dummies = pd.get_dummies(airline_data, columns = cols_to_transform)
df_with_dummies.head(20)


# <hr style="border:2px solid gray"> </hr>

# ### Checking for missing values or NA

# In[25]:


NaN_airline_data = df_with_dummies[df_with_dummies.isnull().any(1)]
NaN_winequality_red_data = winequality_red_data[winequality_red_data.isnull().any(1)]
NaN_parkinsons_upd = parkinsons_upd[parkinsons_upd.isnull().any(1)]


# In[26]:


NaN_airline_data


# In[27]:


NaN_winequality_red_data


# In[28]:


NaN_parkinsons_upd


# ### All three datasets were checked for missing or NaN values but no such values were present.

# <hr style="border:2px solid gray"> </hr>

# ### Splitting Dataset
# 
# <p>The Dataset will be split into 80% Train set and 20% Test using the function below:</p>

# In[49]:


def DataSplit(df,column):
    X_Data = df.drop([column],axis=1)        # Dropping price column from X_Data
    Y_Data= np.array(df[column]).transpose()
    
    X_Data.insert(0,1,np.ones(len(X_Data)))        # Adding Bias
    
    num_of_rows = int(len(X_Data) * 0.8)
    
    X_Train_Data = X_Data.iloc[:num_of_rows,:]        #indexes rows for training data
    X_Test_Data = X_Data.iloc[num_of_rows:,:]      #indexes rows for test data
    
    np.random.shuffle(X_Data.values)  #shuffles data to make it random 
        
    Y_Train_Data = Y_Data[0:num_of_rows]#indexes rows for training data
    Y_Test_Data = Y_Data[num_of_rows:]
    
    return X_Train_Data,Y_Train_Data,X_Test_Data, Y_Test_Data


# <p>This function is used later for splitting all the given datasets. </p>

# <hr style="border:2px solid gray"> </hr>

# ### Part B.) Linear Regression with Real-World Data

# ###  Minimizing loss using gradient descent algorithm

# #### Function to calculate loss

# In[45]:


def loss_function (beta,X_Train,Y_Train):
    # y_hat=X*beta
    y_prediction = np.dot(X_Train,beta)
    loss_function = np.matmul((Y_Train-y_prediction).T,(Y_Train-y_prediction))
    return loss_function


# #### Function to calculate derivative of loss function

# In[46]:


def loss_function_derivative(beta,X_Train,Y_Train):
    # gradient=-2*XT*(Y-X*beta)
    grad_loss = -2 * np.dot(X_Train.T, Y_Train - np.dot(X_Train, beta))
    return grad_loss


# #### Function to calculate RMSE error

# In[47]:


# RMSE on Test Data
def Root_Mean_Square_Error(initial_betas,X_Test_Data,Y_Test_Data):
    RMSE=np.sqrt(np.mean(Y_Test_Data-np.dot(X_Test_Data, initial_betas))**2)
    return RMSE


# #### Minimize loss function using gradient descent
# 
# <p>The loss function to be minimized is the least squares loss. In each iteration the difference between the previous function value and the current function value is calculated and plotted at the end. Also ‘Root Mean Squared Error’ is calculated in each iteration and plotted at the end.</p>

# In[316]:


def minimize_GD(alpha,X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data,Iterations,Title):
    result = []
    rmse=[]
    betas = np.zeros((X_Train_Data.shape[1]))
    for i in range(0,Iterations): 
        loss = Y_Train_Data - np.dot(X_Train_Data, betas)
        betas = betas - alpha * (-2 * np.dot(X_Train_Data.T, loss))
        newLoss = np.sum(np.square(Y_Train_Data - np.dot(X_Train_Data, betas)))
        result.append(np.abs(newLoss - np.sum(np.square(loss))))
        rmse.append(np.sqrt(np.mean((Y_Test_Data - np.dot(X_Test_Data, betas))**2)))
        
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))
    ax[0].set_title(Title)
    ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi)')
    ax[0].plot(result, 'green')
    
    ax[1].set_title(Title)
    ax[1].set(xlabel='No of Iterations', ylabel='RMSE')
    ax[1].plot(rmse, 'green')
    
    
    
    return


# <hr style="border:2px solid gray"> </hr>

# ### Linear Regression on Airfare and Demand Dataset:
# 
# <p>For this dataset I do not drop any columns as the positive and negative correlation with the target column ‘price’ are very small and would not make much difference.</p>

# In[52]:


df_with_dummies.corr()


# #### Splitting the dataset into train and test sets

# In[50]:


target_column="price"
X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data = DataSplit(df_with_dummies, target_column)


# #### Minimizing loss using gradient descent

# <p>If we select <b>imax = 100</b></p>

# In[116]:


step_length=0.00000000001
iterations_to_run = 100
plot_title = "Airfair and Demand Dataset"
minimize_GD(step_length,X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data,iterations_to_run,plot_title)


# <p>If we select <b>imax = 1000</b></p>

# In[117]:


step_length=0.00000000001
iterations_to_run = 1000
plot_title = "Airfair and Demand Dataset"
minimize_GD(step_length,X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data,iterations_to_run,plot_title)


# ##### The loss function has been minimized using ‘Gradient Descent’ algorithm. Above plots show that for 100 iterations the difference between previous and current function values and the RMSE converged later than with 1000 iterations. Therefore, I calculated all my results using 1000 iterations.

# #### For 1000 iterations, the graphs were plotted with three different values of step length.

# <p><b>Step Length = 0.000000000001</b></p>

# In[118]:


step_length=0.000000000001
iterations_to_run = 1000
plot_title = "Airfair and Demand Dataset"
minimize_GD(step_length,X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data,iterations_to_run,plot_title)


# <p><b>Step Length = 0.00000000001</b></p>

# In[119]:


step_length=0.00000000001
iterations_to_run = 1000
plot_title = "Airfair and Demand Dataset"
minimize_GD(step_length,X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data,iterations_to_run,plot_title)


# <p><b>Step Length = 0.0000000001</b></p>

# In[120]:


step_length=0.0000000001
iterations_to_run = 1000
plot_title = "Airfair and Demand Dataset"
minimize_GD(step_length,X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data,iterations_to_run,plot_title)


# ##### It can be seen from all |f(xi−1) − f(xi)| graphs that for larger step length difference is very high at start and it's decreasing quickly with iterations and converges fastly. For smaller step lengths difference error is large and it's taking more time to converge because steps are smaller. 
# 
# ##### RMSE shows the models performance on test set. Which shows that with appropiate step length model learns very quickly
# 
# <p>It is important to note here that for step length greater than 0.0000000001 the error does not converge it diverges.</p>

# <hr style="border:2px solid gray"> </hr>

# ### Linear Regression on Wine Quality Dataset:
# 
# <p>In this dataset, there were no non-numeric columns. Therefore, there was no need to perform one-hot encoding.No columns were dropped as the correlation between the quality column and other columns is very small and the results are similar even after dropping the columns</p>

# In[121]:


winequality_red_data.head(5)


# In[122]:


# winequality_red_data.corr(method='pearson')


# #### Splitting the dataset into train and test sets

# In[123]:


column="quality"
Xtrain_W, Ytrain_W, Xtest_W, Ytest_W = DataSplit(winequality_red_data,column)


# #### step_length=0.0000001

# In[124]:


step_length=0.0000001
iterations_to_run = 1000
plot_title = "Wine Quality Dataset"
minimize_GD(step_length,Xtrain_W, Ytrain_W, Xtest_W, Ytest_W,iterations_to_run,plot_title)


# ##### The difference converged to zero quickly in the first iteration as seen in the above graph. For step length greater than 0.0000001 the difference diverges, leaving very few choices of step lengths

# #### step_length=0.00000001

# In[125]:


step_length=0.00000001
iterations_to_run = 1000
plot_title = "Wine Quality Dataset"
minimize_GD(step_length,Xtrain_W, Ytrain_W, Xtest_W, Ytest_W,iterations_to_run,plot_title)


# #### step_length=0.000000001

# In[126]:


step_length=0.000000001
iterations_to_run = 1000
plot_title = "Wine Quality Dataset"
minimize_GD(step_length,Xtrain_W, Ytrain_W, Xtest_W, Ytest_W,iterations_to_run,plot_title)


# ##### The difference now takes more time to converge due to step length being too small. Same explanation implies to the plot of RMSE using these two step lengths (0.00000001, 0.000000001).

# <hr style="border:2px solid gray"> </hr>

# ### Linear Regression on Parkisons Dataset:
# 
# <p>Similar to the wine quality dataset, there are no non-numeric columns in this dataset. Correlation among the columns was calculated and the correlation of the columns with the target column ‘total_UPDRS’ was very small. No columns were dropped. The results with dropping a few columns and without dropping any columns were compared and they were similar.</p>

# In[129]:


parkinsons_upd.dtypes


# In[130]:


parkinsons_upd.corr(method='pearson')


# #### Splitting dataset into train and test sets

# In[132]:


column="total_UPDRS"
Xtrain_P, Ytrain_P, Xtest_P, Ytest_P = DataSplit(parkinsons_upd,column)


# #### step_length=0.00000001

# In[134]:


step_length=0.00000001
iterations_to_run = 1000
plot_title = "Parkisons Dataset"
minimize_GD(step_length, Xtrain_P, Ytrain_P, Xtest_P, Ytest_P,iterations_to_run,plot_title)


# #### step_length=0.000000001

# In[135]:


step_length=0.000000001
iterations_to_run = 1000
plot_title = "Parkisons Dataset"
minimize_GD(step_length, Xtrain_P, Ytrain_P, Xtest_P, Ytest_P,iterations_to_run,plot_title)


# #### step_length=0.0000000001

# In[317]:


step_length=0.0000000001
iterations_to_run = 1000
plot_title = "Parkisons Dataset"
minimize_GD(step_length, Xtrain_P, Ytrain_P, Xtest_P, Ytest_P,iterations_to_run,plot_title)


# ##### The difference with step length 0.0000000001 takes more time to converge due to step length being too small. It is important to note that with step length higher than 0.00000001, the difference diverges
# 
# ##### RMSE with step length 0.000000001 converges later than RMSE with step length 0.00000001. However, RMSE with step length 0.0000000001 takes more time to converge than both of the above indicating that the step length is too small

# <hr style="border:2px solid gray"> </hr>

# ### Exercise 3.) Steplength Control for Gradient Descent

# ### Utility Functions

# #### Steplength-backtracking

# In[258]:


def stepsize_backtracking(beta,Xtrain,Ytrain):
    
    alpha=0.1
    B= 0.5
    mu = 1
    
    while (loss_function(beta,Xtrain,Ytrain) - loss_function(beta+(mu*-1*loss_function_derivative(beta,Xtrain,Ytrain)),Xtrain,Ytrain)) < (alpha*mu*np.matmul(loss_function_derivative(beta,Xtrain,Ytrain).T, loss_function_derivative(beta,Xtrain,Ytrain))):
        mu = mu * B
    return mu


# In[259]:


def gradientD_BTLS(Xtrain, Ytrain, Xtest, Ytest):
    betas = np.zeros((Xtrain.shape[1]))
    newLoss = Ytrain - np.dot(Xtrain, betas)
    result_BTLS = []
    rmse_BTLS = []
    for i in range(0,1000): 
        mu_i = stepsize_backtracking(betas, Xtrain, Ytrain)
        loss = Ytrain - np.dot(Xtrain, betas)
        betas = betas - mu_i * (-2 * np.dot(Xtrain.T, loss))
        newLoss = np.sum(np.square(Ytrain - np.dot(Xtrain, betas)))
        result_BTLS.append(np.abs(newLoss - np.sum(np.square(loss))))
        rmse_BTLS.append(np.sqrt(np.mean((Ytest - np.dot(Xtest, betas))**2)/len(Xtrain)))
        
        if np.abs(newLoss - np.sum(np.square(loss))) < 0.1:
            break
    return result_BTLS, rmse_BTLS


# #### Steplength-bolddriver

# In[260]:


def boltdriver_steplength(beta,Xtrain,Ytrain,alpha_old,alpha_plus,alpha_minus):
    alpha=alpha_old * alpha_plus
    while (loss_function(beta,Xtrain,Ytrain)-loss_function(beta+(alpha*-1*loss_function_derivative(beta,Xtrain,Ytrain)),Xtrain,Ytrain)<=0):
        alpha=alpha * alpha_minus
    return alpha


# In[264]:


def gradientD_boltdriver(Xtrain, Ytrain, Xtest, Ytest):
    betas = np.zeros((Xtrain.shape[1]))
    newLoss = Ytrain - np.dot(Xtrain, betas)
    result_BTLS = []
    rmse_BTLS = []
    for i in range(0,1000):
        aplha = boltdriver_steplength(betas, Xtrain, Ytrain,1,1.1,0.5)
        loss = Ytrain - np.dot(Xtrain, betas)
        betas = betas - aplha * (-2 * np.dot(Xtrain.T, loss))
        newLoss = np.sum(np.square(Ytrain - np.dot(Xtrain, betas)))
        #print (newLoss, np.sum(np.square(loss)))
        result_BTLS.append(np.abs(newLoss - np.sum(np.square(loss))))
        rmse_BTLS.append(np.sqrt(np.mean((Ytest - np.dot(Xtest, betas))**2)/len(Xtrain)))
        
        if np.abs(newLoss - np.sum(np.square(loss))) < 0.1:
            break
    return result_BTLS, rmse_BTLS


# #### Look-ahead optimizer

# In[341]:


def gradientD_lookAhead(Xtrain, Ytrain, Xtest, Ytest, alpha = 0.5, mu = 0.000000001):
    k=10 #batch size
    betas = np.zeros((Xtrain.shape[1], k)) #fast weights
    slowBetas = np.zeros((Xtrain.shape[1], 1)) # slow weights
    result_BTLS = []
    rmse_BTLS = []
    for i in range(0,500): 
        betas[:,0] = slowBetas.T
        
        # lets the internalized ‘faster’ optimizer explore for k=10 batches.
        # LookAhead then takes the difference between it’s saved weights and GD latest weights once the k interval is hit, 
        # and multiplies that by an alpha param (.5 by default) at every k batches, and updates the weights for GD.
        
        for j in range(1,k): 
            d = int(Xtrain.shape[0]/k)
            choices = np.random.choice(Xtrain.shape[0], d, replace=False)
            X_t = Xtrain.iloc[choices,:]
            Y_t = Ytrain[choices]
            loss = Y_t - np.dot(X_t, betas[:,j-1])
            betas[:,j] = betas[:,j-1] - mu * (-2 * np.dot(X_t.T, loss))
            
        slowBetas =  slowBetas + alpha *(betas[:,k-1].reshape((Xtrain.shape[1],1)) - slowBetas) # update slow weight 
        newLoss = np.sum(np.square(Ytrain - np.dot(Xtrain, slowBetas)))
        result_BTLS.append(np.abs(newLoss - np.sum(np.square(loss))))
        rmse_BTLS.append(np.sqrt(np.mean((Ytest - np.dot(Xtest, slowBetas))**2)/len(Xtrain)))
        
    return result_BTLS, rmse_BTLS


# #### Each one of the above-mentioned algorithms were used in the gradient descent function for calculating the step length in each iteration for all three datasets and results were plotted.

# <hr style="border:2px solid gray"> </hr>

# ### Airfare and Demand Dataset:

# In[329]:


column="price"
X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data = DataSplit(df_with_dummies, column)


# ##### Step length is calculated using the Backtrack line search algorithm in each iteration of the gradient descent for the ‘Airfare and Demand’ dataset. The following results are obtained:

# In[330]:


result_BTLS, rmse_BTLS = gradientD_BTLS(X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data)


# In[331]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using BackTrack Line Search')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Airline_Data')
ax[0].plot(result_BTLS, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using BackTrack Line Search')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BTLS - Airfaire_Data')
ax[1].plot(rmse_BTLS, 'red')


# <p>Using ‘Backtrack line search’ the difference converged very quickly. However, it took some time for the RMSE to start converging to zero. The fluctuations in the plot can be interpreted as the change in step length in every iteration in order to choose the optimal step length by the algorithm</p>

# #### Step length is calculated using the ‘Bold Drivers’ algorithm in each iteration of the gradient descent for the ‘Airfare and Demand’ dataset. The following results are obtained:

# In[265]:


result_boltdriver,rmse_boltdriver = gradientD_boltdriver(X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data)


# In[267]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using boltdriver steplength')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Airline_Data')
ax[0].plot(result_boltdriver, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using boltdriver steplength')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BD - Airfaire_Data')
ax[1].plot(rmse_boltdriver, 'red')


# <p>The difference converged to zero earlier using the ‘Bold Drivers’ algorithm in about 250 iterations. The RMSE also converged faster using the ‘Bold Drivers’ algorithm in about 250 iterations, which is the <b>fastest among the three</b> in the ‘Airfare Demand’ dataset. However, the RMSE started converging after some 100 iterations in ‘Bold Drivers’ algorithm</p>

# #### Step length is calculated using the Look Ahead optimizer in each iteration of the gradient descent for the ‘Airfare and Demand’ dataset. The following results are obtained:

# In[342]:


result_LH, rmse_LH = gradientD_lookAhead(X_Train_Data,Y_Train_Data,X_Test_Data,Y_Test_Data)


# In[343]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using look ahead optimizer')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Airline_Data')
ax[0].plot(result_LH, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using look ahead optimizer')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_LA - Airfaire_Data')
ax[1].plot(rmse_LH, 'red')


# <p>Using look ahead optimizer the difference does not converge to zero in 500 iterations. Also Rmse loss is not converged.I think reason for this behaviour is that I used simple gradient descent as internal optimizer for look ahead optimzer. This exercise is based on gradient descent so I used this algorithm as internal optimzer of look ahead optimzer. But in look ahead optimzer paper, the authors used SGD or Adam as internal optimizer for look ahead. They acheived good results on these internal optimizers. So, Results will be improved if we use Adam or SGD as internal optimizer of look ahead optimizer</p>

# <hr style="border:2px solid gray"> </hr>

# ### Wine Quality Dataset:

# #### Step length is calculated using the ‘Backtrack line search’ algorithm in each iteration of the gradient descent for the ‘Wine Quality’ dataset. The following results are obtained:

# In[299]:


result_BTLS_W, rmse_BTLS_W = gradientD_BTLS(Xtrain_W, Ytrain_W, Xtest_W, Ytest_W)


# In[300]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using BackTrack Line Search')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Wine Quality Red')
ax[0].plot(result_BTLS_W, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using BackTrack Line Search')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BTLS - Wine Quality Red')
ax[1].plot(rmse_BTLS_W, 'red')


# <p>Both the difference and RMSE converged in 120 iterations which is faster than in the gradient descent with fixed step length.</p>

# #### Step length is calculated using the ‘Bold Drivers’ algorithm in each iteration of the gradient descent for the ‘Wine Quality’ dataset. The following results are obtained:

# In[301]:


result_boltdriver_W,rmse_boltdriver_W = gradientD_boltdriver(Xtrain_W, Ytrain_W, Xtest_W, Ytest_W)


# In[302]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using boltdriver steplength')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Wine Quality Red')
ax[0].plot(result_boltdriver_W, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using boltdriver steplength')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BD - Wine Quality Red')
ax[1].plot(rmse_boltdriver_W, 'red')


# <p>Using the ‘Bold Drivers’ algorithm for step length, both the RMSE and the difference converge in 30 iterations. This is faster than the step length from the ‘Back-Track Line Search’ algorithm for the ‘Wine Quality’ dataset.</p>

# #### Step length is calculated using the Look Ahead optimizer in each iteration of the gradient descent for the ‘Airfare and Demand’ dataset. The following results are obtained:

# In[344]:


result_LH_W, rmse_LH_W = gradientD_lookAhead(Xtrain_W, Ytrain_W, Xtest_W, Ytest_W)


# In[345]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using look ahead optimizer')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Wine Quality Red')
ax[0].plot(result_LH_W, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using look ahead optimizer')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BD - Wine Quality Red')
ax[1].plot(rmse_LH_W, 'red')


# <p>Using look ahead optimizer, both the difference and  Rmse loss took more time to converge than previous step length algorithms.I think reason for this behaviour is that I used simple gradient descent as internal optimizer for look ahead optimzer. This exercise is based on gradient descent so I used this algorithm as internal optimzer of look ahead optimzer. But in look ahead optimzer paper, the authors used SGD or Adam as internal optimizer for look ahead. They acheived good results on these internal optimizers. So, Results will be improved if we use Adam or SGD as internal optimizer of look ahead optimizer</p>

# <hr style="border:2px solid gray"> </hr>

# ## Parkisons Dataset:

# ##### Step length is calculated using the ‘Backtrack line search’ algorithm in each iteration of the gradient descent for the ‘Parkinsons’ dataset. The following results are obtained:

# In[306]:


result_BTLS_P, rmse_BTLS_P = gradientD_BTLS(Xtrain_P, Ytrain_P, Xtest_P, Ytest_P)


# In[307]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using BackTrack Line Search')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Parkinsons')
ax[0].plot(result_BTLS_P, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using BackTrack Line Search')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BTLS - Parkinsons')
ax[1].plot(rmse_BTLS_P, 'red')


# <p>Both the difference and the RMSE converge before 400 iterations which is faster than in the gradient descent with fixed step length</p>

# #### Step length is calculated using the ‘Bold Drivers’ algorithm in each iteration of the gradient descent for the ‘Parkinsons’ dataset. The following results are obtained:

# In[310]:


result_boltdriver_P,rmse_boltdriver_P = gradientD_boltdriver(Xtrain_P, Ytrain_P, Xtest_P, Ytest_P)


# In[311]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using BackTrack Line Search')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of Parkinsons')
ax[0].plot(result_boltdriver_P, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using BackTrack Line Search')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BTLS - Parkinsons')
ax[1].plot(rmse_boltdriver_P, 'red')


# <p>Both the difference and the RMSE converge in about 350 iterations. In the ‘Parkinsons’ dataset, the RMSE and difference converged almost on the same iterations using the ‘Bold Drivers’ algorithm for calculating step length for gradient descent when compared with the results using  ‘Back Track Line Search’ algorithm for step length.</p>

# #### Step length is calculated using the Look Ahead optimizer in each iteration of the gradient descent for the ‘Airfare and Demand’ dataset. The following results are obtained:

# In[ ]:


result_LH_P, rmse_LH_P = gradientD_lookAhead(Xtrain_P, Ytrain_P, Xtest_P, Ytest_P)


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

# Plotting epsilon
ax[0].set_title('GD using look ahead optimizer')
ax[0].set(xlabel='No of Iterations', ylabel='f(xi−1) − f(xi) of of Parkinsons')
ax[0].plot(result_LH_P, 'red')
    
# Plotting RMSE

ax[1].set_title('GD using look ahead optimizer')
ax[1].set(xlabel='No of Iterations', ylabel='RMSE_BD - of Parkinsons')
ax[1].plot(rmse_LH_P, 'red')


# <p>Using look ahead optimizer, both the difference and  Rmse loss took more time to converge than previous step length algorithms.I think reason for this behaviour is that I used simple gradient descent as internal optimizer for look ahead optimzer. This exercise is based on gradient descent so I used this algorithm as internal optimzer of look ahead optimzer. But in look ahead optimzer paper, the authors used SGD or Adam as internal optimizer for look ahead. They acheived good results on these internal optimizers. So, Results will be improved if we use Adam or SGD as internal optimizer of look ahead optimizer</p>

# <hr style="border:2px solid gray"> </hr>

# In[ ]:




