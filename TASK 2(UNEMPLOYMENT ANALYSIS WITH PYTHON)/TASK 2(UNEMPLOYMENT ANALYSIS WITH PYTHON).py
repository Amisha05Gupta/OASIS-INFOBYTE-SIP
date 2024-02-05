#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Unemployment in India.csv')
df.head()


# In[4]:


# Data Cleaning
df['Date'] = pd.to_datetime(df['Date'])
df['Date']


# In[5]:


print(df.info())


# In[10]:


# Distribution of unemployment rate over time
plt.figure(figsize=(20, 15))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df, hue='Region')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()


# In[13]:


# Trends in unemployment rate during and after Covid-19
covid_start_date = pd.to_datetime('2020-03-01')
covid_end_date = pd.to_datetime('2022-01-01')
covid_data = df[(df['Date'] >= covid_start_date) & (df['Date'] <= covid_end_date)]
covid_data


# In[17]:


# Unemployment rates in different regions
plt.figure(figsize=(20, 6))
sns.boxplot(x='Region', y='Estimated Unemployment Rate (%)', data=df)
plt.title('Unemployment Rate Across Regions')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=50)
plt.show()


# In[20]:


#Correlation between unemployment rate and other factors
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[46]:


# Temporal Changes in Unemployment Rate Across Regions
plt.figure(figsize=(20, 8))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df, hue='Region', ci=None)
plt.title('Temporal Changes in Unemployment Rate Across Regions')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# In[26]:


# Heatmap
heatmap_data = df.pivot_table(index='Region', columns='Date', values='Estimated Unemployment Rate (%)')
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cbar_kws={'label': 'Unemployment Rate (%)'})
plt.title('Temporal Changes in Unemployment Rate - Heatmap')
plt.show()


# In[39]:


plt.figure(figsize=(20, 6))
sns.barplot(x='Region', y='Unemployment Rate (%)', data=df)
plt.title(f'Unemployment Rate by Region on {df}')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=50)
plt.show()
df.to_csv('cleaned_unemployment_data.csv', index=False)

