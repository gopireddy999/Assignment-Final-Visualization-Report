#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[81]:


# Load the dataset
data = pd.read_csv('D:/data/Python/climate-ds.csv')


# In[82]:


# Initial data overview
print(data.head())


# In[83]:


print(data.describe())


# In[84]:


# Check for missing values in each column
missing_values = data.isnull().sum()


# In[85]:


# Print the number of missing values in each column
print("Missing values in each column:")
print(missing_values)


# In[86]:


# Creating a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[87]:


# Line plot for average rainfall over years
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Year', y='average_rain_fall_mm_per_year')
plt.title('Average Rainfall Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rainfall (mm)')
plt.show()


# In[88]:


# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x='Year', y='average_rain_fall_mm_per_year', hue='Area')
plt.title('Average Rainfall by Year for Different Areas')
plt.xlabel('Year')
plt.ylabel('Average Rainfall (mm)')
plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[89]:


# Create the Area plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Year', y='hg/ha_yield', hue='Area', marker='o')
plt.title('Trend of Crop Yield Over Years by Area')
plt.xlabel('Year')
plt.ylabel('Crop Yield (hg/ha)')
plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[90]:


# Boxplot of Crop Yield Distribution by Crop Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Item', y='hg/ha_yield')
plt.xticks(rotation=45)
plt.title('Crop Yield Distribution by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Crop Yield (%)')
plt.show()


# In[91]:


# Scatter Plot of Crop Yield vs. Average Rainfall
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='average_rain_fall_mm_per_year', y='hg/ha_yield')
plt.title('Crop Yield vs. Average Rainfall')
plt.xlabel('Average Rainfall (mm/year)')
plt.ylabel('Crop Yield (%)')
plt.show()


# In[61]:


# Histogram of Pesticides Usage Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='pesticides_tonnes', kde=True, color='green')
plt.title('Distribution of Pesticides Usage')
plt.xlabel('Pesticides (Tonnes)')
plt.ylabel('Frequency')
plt.show()


# In[62]:


# Scatter plot
plt.figure(figsize=(10, 6))
# Creating a scatter plot where color depends on 'avg_temp'
scatter = plt.scatter(data=data, x='average_rain_fall_mm_per_year', y='hg/ha_yield', 
                      c=data['avg_temp'], cmap='coolwarm', alpha=0.6)
plt.title('Crop Yield vs. Rainfall')
plt.xlabel('Average Rainfall (mm/year)')
plt.ylabel('Crop Yield (hg/ha)')
# Creating a colorbar
plt.colorbar(scatter, label='Average Temperature (°C)')

plt.grid(True)
plt.tight_layout()
plt.show()


# In[63]:


# Pairplot of Crop Yield, Rainfall, Temperature, and Pesticides Usage 
plt.figure(figsize=(10, 8))
sns.pairplot(data=data, vars=['hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes'], hue='Area', palette='Set1')
plt.suptitle('Pairplot of Crop Yield, Rainfall, Temperature, and Pesticides Usage')
plt.show()


# In[64]:


# FacetGrid of Scatter Plots
numerical_vars = ['average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes']
g = sns.PairGrid(data, y_vars=['hg/ha_yield'], x_vars=numerical_vars, height=4)
g.map(sns.scatterplot)
g.set(ylim=(0, None))  # Adjust y-axis limit for better visualization
plt.suptitle('Scatter Plots of Crop Yield vs. Climate and Pesticides Usage', y=1.05)
plt.show()


# In[70]:


# Data description
summary_stats = data.describe()

# Transpose the summary statistics DataFrame for better readability
summary_stats = summary_stats.T

# Display the summary statistics table
print(summary_stats)


# In[71]:


# Create a pivot table to show average crop yield for each crop type and year
pivot_table = data.pivot_table(index='Item', columns='Year', values='hg/ha_yield', aggfunc='mean')

# Display the pivot table
print(pivot_table)


# In[72]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# In[73]:



# Encoding categorical features
label_encoder = LabelEncoder()
data['Area'] = label_encoder.fit_transform(data['Area'])
data['Item'] = label_encoder.fit_transform(data['Item'])


# In[74]:


# Splitting the dataset into features (X) and target variable (y)
X = data.drop(columns=['hg/ha_yield'], axis=1)
y = data['hg/ha_yield']


# In[75]:


# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[76]:


# Initializing and training the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[77]:


# Making predictions on the test set
y_pred = model.predict(X_test)


# In[78]:


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")


# In[79]:


# Visualizing actual vs. predicted crop yields
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs. Predicted Crop Yields')
plt.xlabel('Actual Crop Yields (hg/ha)')
plt.ylabel('Predicted Crop Yields (hg/ha)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




