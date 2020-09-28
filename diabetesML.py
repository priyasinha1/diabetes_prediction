#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('diabetes.csv')


# In[3]:


data.shape


# In[4]:


data.head(10)


# In[5]:


data.isnull().values.any()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(data[top_corr_features].corr(),annot= True, cmap = "RdYlGn")


# In[7]:


data.corr()


# In[8]:


diabetes_true_count = len(data.loc[data['Outcome'] == True])
diabetes_false_count = len(data.loc[data['Outcome'] == False])
(diabetes_true_count,diabetes_false_count)


# In[9]:


data[data["Outcome"]==True]["Outcome"].sum()


# In[10]:


from sklearn.model_selection import train_test_split
feature_coloumn = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
predicted_class = ['Outcome']


# In[11]:


X = data[feature_coloumn].values
y = data[predicted_class].values
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state= 10)


# In[ ]:





# In[15]:


print("total no.of rows : {0}".format(len(data)))
print("total no. of missing rows of Pregnancies : {0}".format(len(data.loc[data['Pregnancies']==0])))
print("total no. of missing rows of Glucose : {0}".format(len(data.loc[data['Glucose']==0])))
print("total no. of missing rows of BloodPressure : {0}".format(len(data.loc[data['BloodPressure']==0])))
print("total no. of missing rows of SkinThickness : {0}".format(len(data.loc[data['SkinThickness']==0])))
print("total no. of missing rows of Insulin: {0}".format(len(data.loc[data['Insulin']==0])))
print("total no. of missing rows of BMI: {0}".format(len(data.loc[data['BMI']==0])))
print("total no. of missing rows of DiabetesPedigreeFunction: {0}".format(len(data.loc[data['DiabetesPedigreeFunction']==0])))
print("total no. of missing rows of Age: {0}".format(len(data.loc[data['Age']==0])))


# In[19]:


from sklearn.preprocessing import Imputer
fill_values = Imputer(missing_values = 0, strategy = 'mean',axis = 0)
X_train = fill_values.fit_transform


# In[20]:


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state = 10)
random_forest_model.fit(X_train, y_train.ravel())


# In[18]:


predict_train_data = random_forest_model.predict(X_test)
from sklearn import metrics
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# In[ ]:


model=LogisticRegression(class_weight={1:1.1})
model.fit(data_cleaned[features],data_cleaned['classe'])
X_test = test_data[features]
prediction = model.predict(X_test)
test_data['classe'] = prediction
test_data.loc[:,['id', 'classe']].to_csv('submission.csv', encoding='utf-8', index=False)

