#!/usr/bin/env python
# coding: utf-8

# In[1]:


#lets first start with importing all linear regression library to usen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#lets import the csv data to 2d dataframe
data = pd.read_csv('Salary_Data.csv')


# In[7]:


#Check Pandas Dataframe
data.head()


# In[29]:


#Now lets devide the dataset into Dependent(Varibles) and Independent(Target)
#X is the traning dependent varibles (note here .values is numpy operation)
x = data.iloc[:,:-1].values
x


# In[30]:


#Target Set /Independent Set
y = data.iloc[:,-1:].values
y


# In[48]:


#Till now we have splited the Dataset into X and Y now lets import the Ski-learnt Library to perform spliting into traning and testing 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#above will spilit the dataset into 80% for traning and 20% testing


# In[55]:


print('The total size of data is ',len(data))
print('X traning data and X testing datas are ',len(x_train),'and', len(x_test))
print('Y target data and Y testing datas are ',len(y_train),'and', len(y_test))


# In[56]:


#The best fit model for X and Y types data is Linear Regression
#Lets Import the Linear regression 
from sklearn.linear_model import LinearRegression


# In[57]:


#We have to make a regression object "reg" contains all function of Linear Regression imported module
reg = LinearRegression()


# In[59]:


#Lets train the model using x and y testing and target varible for traning
reg.fit(x_train,y_train)


# In[61]:


#Lets check out that the tranined model predicitng well 
y_prediction = reg.predict(x_test)
x_prediction = reg.predict(x_train)


# In[63]:


#Lets try to plot and visualize the data
plt.scatter(x_train,y_train,color="green")
plt.show()


# In[65]:


#Lets try to put the regression model line from plot and visualize the data
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,x_prediction)
plt.show()


# In[70]:


#Lets check the above regression plot works on testing dataset
print("Test set")
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,x_prediction)
plt.show()


# In[ ]:


#From above diagram we can see and observe its good fit

