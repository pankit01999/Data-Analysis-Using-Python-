#!/usr/bin/env python
# coding: utf-8

# # Title :Uber Data Analysis
# ### 1.Dataset Details
# This dataset holds information related to a user's Uber ride history. Here's a breakdown of the dataset:
# 
# Start Date
# End Date
# Start Location
# End Location
# Miles Driven
# Purpose of the ride (categorized as Business, Personal, Meals, etc.)
# ### 2.Objective
# The objective is to gain insights and understand the travel behavior of a typical Uber customer.

# # 3.Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os


# # 4.Import Dataset 

# In[2]:


df = pd.read_csv('My Uber Drives - 2016.csv')
df.head()


# In[3]:


print(df.shape)
df.dtypes


# ### There are 6 catagorical vars and 1 numeric type variable Here STATR_DATE and
# ### END_DATE* are in object type. We need to convert them back into datetime variable*
# ### 5 Checking for null values

# In[4]:


df.isna().sum()


# In[5]:


df[df['END_DATE*'].isna()]


# #### As we can see this row contains wrong data for most of the columns. We will delete it.
# #### Dropping row containing null values

# In[6]:


df.drop(df[df['END_DATE*'].isna()].index,axis=0,inplace=True)


# In[7]:


df.isna().sum()


# In[8]:


df.info()


# ### Now we have null data only in Purpose column. As we have more than 55% data missing. So I am dropping this columns and excluding this from this analysis.You may also delete the null value rows and include this column in the analysis.
# ### sns.countplot(df['PURPOSE*'], order=df['PURPOSE*'].value_counts().index)

# In[9]:


# droppig Purpose
df.drop(['PURPOSE*'],axis=1,inplace=True)


# In[10]:


df.head(2)


# ### 6 Checking for duplicate rows

# In[11]:


df[df.duplicated()]


# In[12]:


### We will remove this duplicate row
df.drop(df[df.duplicated()].index, axis=0, inplace=True)
df[df.duplicated()]


# In[13]:


### Converting start_date & end_date cols into datetime
df['START_DATE*'] = pd.to_datetime(df['START_DATE*'], format='%m/%d/%Y %H:%M')
df['END_DATE*'] = pd.to_datetime(df['END_DATE*'], format='%m/%d/%Y %H:%M')
df.dtypes


# ### 7 EDA
# ### 8 Univariate
# ### 8.1 1. Category

# In[14]:


df['CATEGORY*'].unique()


# In[15]:


### There are 2 ride-categories… Business: For work related & Personal: For personal travel
df[['CATEGORY*','MILES*']].groupby(['CATEGORY*']).agg(tot_miles=('MILES*','sum'))


# In[16]:


plt.figure()
df[['CATEGORY*','MILES*']].groupby(['CATEGORY*']).agg(tot_miles=('MILES*','sum')).plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Total Miles')
plt.title('Total Miles per Category')


# ### User mainly uses Uber cabs for its Business purposes * Around 94% miles was consumed during Business trips. * Only 6% miles were consumed during personal trips.
# ### 8.2 START*

# In[17]:


len(df['START*'].unique())


# In[18]:


# Top 10 Start places
df['START*'].value_counts(ascending=False)[:10]


# In[19]:


df['START*'].value_counts(ascending=False)[:10].plot(kind='barh',ylabel='Places',xlabel='Pickup Count',title='Top 10 Pickup places')


# ### Cary is the most popular Starting point for this user
# ### 8.3 STOP*

# In[20]:


len(df['STOP*'].unique())


# There are 188 unique Drop points (destination)

# In[21]:


df['STOP*'].value_counts(ascending=False)[:10].plot(kind='barh', ylabel='Places', xlabel='Dropoff Count', title='Top 10 Dropoff Places')


# Cary is the most popular Stop place for this user. Maybe his home is in Cary (as
# mostly start & stop are from here)

# In[23]:


df[df['START*']=='Unknown Location']['START*'].value_counts()


# In[24]:


df[df['STOP*']=='Unknown Location']['STOP*'].value_counts()


# ### 8.4 MILES*
# 

# In[25]:


sns.histplot(df['MILES*'],kde=True)


# In[26]:


### Miles data is Rightly Skewed
df.describe().T


# #### 8.5 Multivariate analysis
# 

# In[27]:


df.head()


# In[28]:


df.groupby(['START*','STOP*'])['MILES*'].apply(print)


# In[29]:


df.groupby(['START*', 'STOP*'])['MILES*'].sum().sort_values(ascending=False)[1:11]


# Cary-Durham & Cary-Morrisville and vice versa are the farthest distance ride.
# Checking for Round Trip

# In[30]:


def is_roundtrip(row):
    if row['START*'] == row['STOP*']:
        return 'YES'
    else:
        return 'NO'

df['ROUND_TRIP*'] = df.apply(is_roundtrip, axis=1)
sns.countplot(x='ROUND_TRIP*', data=df, order=df['ROUND_TRIP*'].value_counts().index)


# In[31]:


df['ROUND_TRIP*'].value_counts()


# User mostly take single-trip Uber rides. * Around 75% trip is single-trip and 25% are Round-Trip
# ### 8.6 Calculating Ride duration

# In[32]:


df.dtypes


# In[33]:


df['Ride_duration'] = df['END_DATE*'] - df['START_DATE*']


# In[34]:


df.head()


# Converting Ride_duration into Minutes This is a Python lambda function that takes a single argument “x”.
# The function first calls the to_pytimedelta() method on pd.Timedelta, which converts the input x into a datetime.timedelta object.
# The function then calculates the total number of minutes in the timedelta object, which is done by first getting the number of days using the days attribute and multiplying it by 24 hours and
# 60 minutes per hour. Then, the number of seconds is divided by 60 to convert them into minutes, and added to the previously calculated number of minutes. The final result is the total number of
# minutes in the timedelta object.
# This function could be used to calculate the duration of a time interval in minutes, which could be useful in a variety of applications such as analyzing time-series data or calculating the length of
# time between two events.
# Note that this function assumes that the input x is a valid pd.Timedelta object and may raise errors if the input is not in the expected format.

# In[40]:


df['Ride_duration'] = pd.to_timedelta(df['Ride_duration'])

# Now perform the conversion to minutes
df['Ride_duration'] = df['Ride_duration'].apply(lambda x: x.total_seconds() // 60)
df.head()


# In[41]:


df['month'] = pd.to_datetime(df['START_DATE*']).dt.month
df['Year'] = pd.to_datetime(df['START_DATE*']).dt.year
df['Day'] = pd.to_datetime(df['START_DATE*']).dt.day
df['Hour'] = pd.to_datetime(df['START_DATE*']).dt.hour
df['day_of_week'] = pd.to_datetime(df['START_DATE*']).dt.dayofweek
days = {0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
df['day_of_week'] = df['day_of_week'].apply(lambda x: days[x])
df.head()


# In[42]:


df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])
df.head()


# In[43]:


### Total rides/month
print(df['month'].value_counts())


# In[44]:


sns.countplot(x='month',data=df,order=pd.value_counts(df['month']).index,hue='CATEGORY*')


# Most number of rides were in month of December (all of them were Business trips) Top
# 5 months having most trips were: December,August,November,February & March. Uber Ride
# was used at Feb,Mar,Jul,Jun & Apr for personal trips.

# In[45]:


sns.countplot(x='day_of_week',data=df,order=pd.value_counts(df['day_of_week']).index,hue='CATEGORY*')


# FRIDAY was the day at which uber rides were mostly used
# Average distance covered/month

# In[46]:


df.groupby('month').mean()['MILES*'].sort_values(ascending = False).plot(kind='bar')
plt.axhline(df['MILES*'].mean(), linestyle='--', color='red', label='Mean␣distance')
plt.legend()
plt.show()


# In[47]:


### User’s Longest ride were on April & shortest were on November
sns.countplot(x='Hour',data=df,order=pd.value_counts(df['Hour']).index,hue='CATEGORY*')


# Maximim number of trips were on Evening & at noon.
# ### 8.6.1 Calculating Trip speed

# In[48]:


df.head()


# In[49]:


df['Duration_hours'] = df['Ride_duration']/60
df['Speed_KM'] = df['MILES*']/df['Duration_hours']
df.head(2)


# In[50]:


fig, ax = plt.subplots()
sns.histplot(x='Speed_KM',data=df,kde=True,ax=ax)


# ### 9 Conclusion
# • User mainly uses Uber cabs for its Business purposes
# – Around 94% miles was consumed during Business trips.
# – Only 6% miles were consumed during personal trips.
# • There are 177 unique starting points
# – Cary is most poplular starting point for this driver.
# • There are 188 unique Stop points.
# – Cary is most poplular drop point for this driver.
# • Cary-Durham & Cary-Morrisville and vice versa are the User’s longest distance
# Uber ride.
# • User usually takes single-trip Uber rides.
# – Around 75% trip is single-trip and 25% are Round-Trip.
# • User’s Most number of rides were in month of December & Least were in September.
# • Friday has maximum number of trips.
# • Afternoons and evenings seem to have the maximum number of trips.
# • User’s Longest ride were on April & shortest were on November

# In[ ]:




