#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import nltk


# In[66]:


#load dataset
df=pd.read_csv('train.csv', encoding='unicode_escape')
df


# In[67]:


#remove unnecessary columns
df=df.drop(['textID', 'text', 'Time of Tweet', 'Age of User', 'Country'], axis=1)
df=df.drop(['Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'], axis=1)
df


# In[68]:


#assign negative as -1 and neutral or positive as 1
df['sentiments']=np.where(df['sentiment']=='negative',-1, 1)
df


# In[69]:


#make text in lowercase
df['selected_text']=df['selected_text'].str.lower()
df


# In[70]:


#make all text as string
stop_words = set(nltk.corpus.stopwords.words('english'))
# df['selected_text'] = df['selected_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df['selected_text'].astype(str)


# In[71]:


#import some sklearn libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# In[72]:


#transformation of the dataset using CountVectorizer
vec=CountVectorizer()
df=df.dropna()
x=vec.fit_transform(df['selected_text'])


# In[73]:


#splitting dataset into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(x, df['sentiments'], test_size=0.2, random_state=42)


# In[74]:


#applying LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)


# In[75]:


#predicting model
y_pred = lr.predict(x_test)


# In[76]:


#calculating accuracy, precision, recall and f1 score
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)


# In[77]:


#print the calculated data
print(f"Accuracy: {round(accuracy*100,2)}%")
print(f"Precision: {round(precision*100,2)}%")
print(f"Recall: {round(recall*100,2)}%")
print(f"F1 Score: {round(f1*100,2)}%")


# In[78]:


#plotting the data usind matplotlib.pyplot
import matplotlib.pyplot as plt
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy, precision, recall, f1])
plt.title('Performance Metrics of Logistic Regression Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()


# In[ ]:




