#!/usr/bin/env python
# coding: utf-8

# In[197]:


import numpy as np
import pandas as numpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[198]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head(5)


# In[199]:



stats = df.describe(include = 'all')
print(stats)


# In[200]:



# Step 1 - Make a scatter plot with square markers, set column names as labels

def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )

    
    # Show column labels on the axes'

    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_yticklabels(y_labels)
    
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
columns = ['id','gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
#columns = ['gender', 'ever_married', 'work_type', 'Residence_type','smoking_status']
#columns = ['age','avg_glucose_level','bmi','heart_disease','id','stroke']
corr = df[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[201]:



sns.pairplot(df, vars=['hypertension', 'age', 
                       'avg_glucose_level','bmi','heart_disease'],
                        kind='reg', hue='stroke')  


# In[202]:


sns.scatterplot(data= df,x = 'bmi',y = 'age',hue = 'stroke')


# In[203]:


sns.countplot(data=data,x='stroke')


# In[204]:



sns.countplot(data=df,x='smoking_status',hue='stroke')


# In[205]:



sns.countplot(data=df,x='ever_married', hue = 'stroke')


# In[206]:



sns.countplot(data=df,x='Residence_type', hue = 'stroke')


# In[207]:


sns.countplot(data=df,x='work_type', hue = 'stroke')


# In[ ]:




