#!/usr/bin/env python
# coding: utf-8

# # Description

# Understand rolling window functions
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

# # Imports

# In[12]:


get_ipython().run_line_magic('load_ext', 'blackcellmagic')


# In[1]:


import pandas as pd


# # Fixed number of rows per rolling window

# In[3]:


df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
df


# In[5]:


get_ipython().run_line_magic('pinfo2', 'df.rolling')


# In[9]:


df.assign(win=lambda x: x.rolling(window=2, win_type='triang').sum())


# In[8]:


df.assign(win=lambda x: x.rolling(window=2, win_type='gaussian').sum(std=3))


# In[10]:


df.assign(win=lambda x: x.rolling(window=2, win_type=None).sum())


# In[11]:


df.assign(win=lambda x: x.rolling(window=2, min_periods=1, win_type=None).sum())


# # Time windows

# A ragged (meaning not-a-regular frequency), time-indexed DataFrame

# In[27]:


df = pd.DataFrame(
    {"B": [1, 1, 2, np.nan, 4]},
    index=[
        pd.Timestamp("20130101 09:00:00"),
        pd.Timestamp("20130101 09:00:01"),
        pd.Timestamp("20130101 09:00:05"),
        pd.Timestamp("20130101 09:00:09"),
        pd.Timestamp("20130101 09:00:10"),
    ],
)


# In[28]:


df


# In[29]:


df.rolling('2s').sum()


# In[30]:


df.assign(win=lambda x: x.rolling('2s').sum())


# In[31]:


df.dtypes


# In[32]:


df.index.dtype


# # Index of first row of window

# If I have a df and I have rolling time windows, I want to know the index of the first row in the window for each row.
# 
# I will use this outside of pandas to do some aggregates.

# In[64]:


tmp = (df
 .assign(idx=lambda x: range(len(x)))
 .assign(win=lambda x: x.rolling('2s')['B'].sum(),)
 .assign(win_min_idx=lambda x: x.rolling('2s')['idx'].min(),
         win_max_idx=lambda x: x.rolling('2s')['idx'].max())
)
tmp


# In[ ]:




