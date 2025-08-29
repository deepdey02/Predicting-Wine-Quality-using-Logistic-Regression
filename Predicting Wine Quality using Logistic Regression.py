#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import OOPs_deep_stats as deep 
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("matplotlib version:", matplotlib.__version__)
print("seaborn version:", sns.__version__)
print("warnings is a standard library module → version not applicable")
print("OOPs_deep_stats is your custom module → version depends on your implementation")


# In[ ]:


df=pd.read_csv("WineQT.csv")
df


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop('Id', axis=1,inplace=True)


# In[ ]:


df.columns


# In[ ]:


sns.pairplot(df,hue='quality')


# In[ ]:


sns.pairplot(df[['fixed acidity','volatile acidity','citric acid','residual sugar','quality']],hue='quality')


# In[ ]:


sns.pairplot(df[['chlorides','free sulfur dioxide','total sulfur dioxide','density','quality']],hue='quality')


# In[ ]:


sns.pairplot(df[['pH','sulphates','alcohol','quality']],hue='quality')


# In[ ]:


x=df.drop('quality',axis=1)
y=df['quality']>=6


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100,stratify=y)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)            # we need to have more iterations
lr.fit(x_train,y_train)


# In[ ]:


y_pred=lr.predict(x_test)
y_pred


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


import joblib         # it is used to save the model
joblib.dump(lr,'wine quality prediction model 76 f1 29-08-25.pkl')

