#!/usr/bin/env python
# coding: utf-8

# # Analyzing PUBG Data with Python
# ## PlayerUnknown's Battlegrounds (PUBG) has taken the gaming world by storm, offering an immersive battle royale experience. Beyond its gaming aspect, PUBG also provides a wealth of data that can be mined and analyzed to gain insights into player behavior,strategies, and performance.
# ## In this Jupyter Notebook project, we will delve into the exciting world of PUBG data analysis using Python. We will be working with a dataset containing a treasure trove of information,including player statistics, match details, and in-game events. Our goal is to harness the power of Python libraries such as Pandas, NumPy, and Matplotlib to extract meaningful insights from this dataset.
# 
# # 1.Import Libraries 

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # 2.Upload CSV File

# In[2]:


df = pd.read_csv('Pubg_Stats.csv')


# # Data Preprocessing
# 
# # a).head()
# 
# head is used show to the By default = 5 rows in the dataset

# In[3]:


df.head()


# # b).tail()
# 
# tail is used to show rows by Descending order

# In[4]:


df.tail()


# # c).shape
# It show the total no of rows & Column in the dataset

# In[5]:


df.shape


# # d).Columns
# It Shows the no of each column

# In[6]:


df.columns


# # e).dtypes
# 
# This Attribute show the data type of each column

# In[7]:


df.dtypes


# # f).unique()
# In a column, It show the unique value of specific column.

# In[8]:


df["Player_Name"].unique()


# # g).nuique()
# It will show the total no of unque value from whole data frame

# In[9]:


df.nunique()


# # h).describe()
# It show the Count, mean , median etc

# In[10]:


df.describe()


# # i).value_counts
# It Shows all the unique values with their count

# In[11]:


df["Player_Name"].value_counts()


# # j).isnull()
# It shows the how many null values

# In[12]:


df.isnull()


# In[13]:


# Various Plots
# 1.HeatMap


# In[14]:


sns.heatmap(df.isnull())
plt.show()


# In[15]:


df.isna().sum()


# # Drop the Unnamed Column

# In[16]:


df.drop(['Unnamed: 0'], axis=1, inplace=True)


# # Show the Rank in Barplot

# In[17]:


df.Rank.value_counts().plot(kind = "bar")


# # Top 10 players By Matches Played

# In[18]:


# Sort the DataFrame by 'Matches_Played' column in descending order and select the top 10 rows
top_10_players = df.sort_values(by='Matches_Played', ascending=False).head(10)
fig = px.bar(top_10_players, x="Player_Name", y="Matches_Played", title="Top 10 Players")
# Customizing colors
fig.update_traces(marker_color='skyblue')
# Show the plot
fig.show()


# # Top 5 players By Matches Played

# In[19]:


# Sort the DataFrame by 'Matches_Played' column in descending order and select the top 5 rows
top_5_players = df.sort_values(by='Matches_Played', ascending=False).head(5)
fig = px.bar(top_5_players, x="Player_Name", y="Matches_Played", title="Top 5 Players")
# Customizing colors to orange
fig.update_traces(marker_color='orange')
# Show the plot
fig.show()


# In[20]:


# Top 5 players By Kills


# In[21]:


# Sort the DataFrame by "Kills" in descending order
df = df.sort_values(by="Kills", ascending=False)

# Select the top 5 players with the highest kills
top_5_players = df.head(5)

# Create a bar plot for the top 5 players
plt.figure(figsize=(10, 6))

plt.bar(top_5_players["Player_Name"], top_5_players["Kills"], color='orange')  # Color bars orange
plt.xlabel("Player Name")
plt.ylabel("Kills")
plt.title("Top 5 Players by Kills")
plt.xticks(rotation=45)
plt.show()


# # Top 5 players By Wins

# In[22]:


# Sort the DataFrame by "Wins" in descending order
df = df.sort_values(by="Wins", ascending=False)
# Select the top 5 players with the highest wins
top_5_players = df.head(5)
# Create a color palette with a mix of orange and red
colors = ['orange', 'red', 'orange', 'red', 'orange']
# Create a bar plot for the top 5 players with custom colors
plt.figure(figsize=(10, 6))
plt.bar(top_5_players["Player_Name"], top_5_players["Wins"], color=colors)
plt.xlabel("Player Name")
plt.ylabel("Wins")
plt.title("Top 5 Players by Wins")
plt.xticks(rotation=45)
plt.show()


# In[23]:


# Player Name By Rank 
# Sort the DataFrame by "Matches Played" in descending order
df = df.sort_values(by="Matches_Played", ascending=False)
# Select the top 5 players with the highest matches played
top_5_players = df.head(5)
plt.figure(figsize=(8, 6))
plt.bar(top_5_players["Player_Name"], top_5_players["Rank"])
plt.xlabel('Player Name')
plt.ylabel('Rank')
plt.title('Players by Rank')
plt.show()


# In[24]:


# How many times Rank are Published 
rank_counts = df["Rank"].value_counts().reset_index()
rank_counts.columns = ["Rank", "Count" ]
rank_counts = rank_counts.sort_values(by="Count", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(rank_counts["Rank"], rank_counts["Count"])
plt.xlabel("Rank")

plt.ylabel("Count")

plt.title("Rank Distribution Among Players")
plt.xticks(rotation=45)

plt.show()


# In[25]:


# How many Times Rank Published by Player Name
# Create a cross-tabulation to count how many times each rank was achieved by ¢
cross_tab = pd.crosstab(df["Player_Name"], df["Rank"]).head(20)
# PLot the bar chart
cross_tab.plot(kind="bar", stacked=True, figsize=(15, 8))
# Customize the plot

plt.xlabel("Player Name")

plt.ylabel("Count")
plt.title("Rank Achievements by Player")

# Show the plot
plt.show()


# In[ ]:




