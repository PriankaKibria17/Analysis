#!/usr/bin/env python
# coding: utf-8

# In[19]:


#imports
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #adjusted the configuration of the plots 

#read in data

df = pd.read_csv('/Users/pri/Downloads/movies.csv')


# In[37]:


df.head() #glimpse of my data

#analyzing the revenue of films


# In[31]:


# allocating missing data
#looping through each column to search for any missing values via %

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round (pct_missing*100)))
    


# In[55]:


#looking for duplicates
df.duplicated().sum() 


# In[46]:


#replacing the missing values for 'budget' , 'gross' with the mean 

mean_budget = df['budget'].mean()
df['budget'].fillna(value=df['budget'].mean(), inplace = True)

gross_mean = df['gross'].mean()
df['gross'].fillna(value=df['gross'].mean(), inplace = True)


# In[47]:


#Data types
df.dtypes


# In[34]:


#changing budget to int type
df['budget']  = df['budget'].astype('int64')
df['gross']  = df['gross'].astype('int64')


# In[54]:


#sorting the data & taking a look 

df.sort_values(by=['gross'], inplace = False, ascending = False)


# In[52]:


pd.set_option('display.max_rows', None) #option to see the whole dataset


# In[56]:


#visualizing the gross and data with gross and budget only for correlation

sns.regplot(x="gross", y="budget", data = df)


# In[57]:


sns.regplot(x="score", y="gross", data = df)


# In[58]:


#correlation matrix between all numeric columns
df.corr(method = 'pearson')


# In[59]:


df.corr(method = 'kendall')


# In[60]:


df.corr(method = 'spearman')


# In[61]:


#correlation matrix

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot = True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()


# In[63]:


#Factorization (assigns a random numeric value for each unique categorical value)

df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[64]:


#correlation matrix for movies//heatmap

correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation Matrix for Movies")

plt.xlabel("Movie Features")

plt.ylabel("Movie Features")

plt.show()


# In[ ]:




