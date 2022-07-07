#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.pandas.set_option("display.max_columns",None) ## Display all the columns if there millions


# In[5]:


dataset=pd.read_csv(r"E:\CODEZ\Data Science\Data Sets\house_prediction_train.csv")


# In[8]:


dataset.shape


# In[9]:


dataset.head()


# ##### In Data Analysis We will Analze to Find out the below stuff
# 1.Missing Values
# 
# 2.All the Numerical Values
# 
# 3.Distribution of the Numerical Values
# 
# 4.Categorical Variables
# 
# 5.Cardibility of Categorical Variables
# 
# 6.Outliers
# 
# 7.Relationship between independent and dependent features

# # 1.Missing Values

# In[27]:


dataset["LotFrontage"].isnull().sum() ## hOW MANY Non Values in that column. Total


# In[34]:


dataset["LotFrontage"].count() ## Total number of Not null values


# In[35]:


np.round(256/1201,5)


# In[37]:


dataset["LotFrontage"].notnull().sum()


# In[40]:


dataset["LotFrontage"].mean()


# In[44]:


np.round(dataset["LotFrontage"].isnull().mean(),4) ##all non values equals to 1. then it devides by total data.. 259 non values/1460 columns


# df.isnull()
# 
# #Mask all values that are NaN as True
# 
# df.isnull().mean()
# 
# #compute the mean of Boolean mask (True evaluates as 1 and False as 0)
# 
# df.isnull().mean().sort_values(ascending = False)
# 
# #sort the resulting series by column names descending
# 
# 
# That being said a column that has values:
# 
# [np.nan, 2, 3, 4]
# is evaluated as:
# 
# [True, False, False, False]
# interpreted as:
# 
# [1, 0, 0, 0]
# Resulting in:
# 
# 0.25

# In[45]:


features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

for features in features_with_na:
    print(features,np.round(dataset[features].isnull().mean(),4), "%  Missing Values ")


# # Since they are many missing values, we need to find the relationship between missing values and Sales Price.If there is not a relationship, we can drop that column

# In[101]:


for features in features_with_na:
    data = dataset.copy()
    
    data[features] = np.where(data[features].isnull(),1 , 0) #if data is null then its 1 and otherwise its zero
    
    data.groupby(features)['SalePrice'].median().plot.bar() ## Features anuwa group krnna. ethokot features walat adalaw sales prices wala median enawa. Non values wala median and Not non values wala median eka ganna pluwn
    plt.title(features)
    plt.ylabel("Median House Price")
    plt.show()


# In[86]:


check=dataset.loc[dataset["LotFrontage"].isnull(),"SalePrice"]


# In[87]:


check.median() ## Lost Frontage null values wala salePrices wala Median eka


# In[88]:


checks=dataset.loc[dataset["LotFrontage"].notnull(),"SalePrice"]


# In[89]:


checks.median()


# In[90]:


dataset["GarageQual"].isnull().sum()


# # We can not remove any features. Because every feature is doing a major Role

# Here With the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section
# 
# From the above dataset some of the features like Id is not required

# # Numerical Variables

# In[100]:


numerical_variables=[features for features in dataset.columns if dataset[features].dtypes !="O"] #O means Object.It means we select features are not objects/ not categorical/not string
print("Number of Numerical Variables:",len(numerical_variables)) #check how many numerical variables are in our list
dataset[numerical_variables].head()


# In[ ]:




