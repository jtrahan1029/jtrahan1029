#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# In[75]:


df = pd.read_csv('final_project_churn.csv')
df.head()


# In[76]:


df.gender.dtype


# In[77]:


for col in df.columns:
    if df[col].dtype == 'O':
        df[col] = df[col].astype('category').cat.codes


# In[78]:


df.dtypes


# In[118]:


df.isnull().sum()


# In[115]:


summary_churn = df.groupby('Churn')
summary_churn.mean()


# In[128]:


import seaborn as sns

f = plt.plot(figsize = (15,6))
sns.distplot(df.MonthlyCharges, kde = True, color = 'blue').set_title('charges')


# In[131]:


plt.plot(figsize = (15,4))
sns.distplot(df.tenure, kde = True, color = 'blue').set_title('charges')


# In[129]:


f = plt.plot(figsize = (15,4))
p = sns.countplot(x = 'gender', data = df)


# In[134]:


f = plt.plot(figsize = (15,4))
p = sns.countplot(x = 'Dependents', data = df)


# In[132]:


f = plt.plot(figsize = (15,4))
p = sns.countplot(x = 'Contract', data = df)


# In[135]:


f = plt.plot(figsize = (15,4))
p = sns.countplot(x = 'gender', data = df, hue = 'Churn')


# In[136]:


f = plt.plot(figsize = (15,4))
p = sns.countplot(x = 'Dependents', data = df, hue = 'Churn')


# In[137]:


f = plt.plot(figsize = (15,4))
p = sns.countplot(x = 'Contract', data = df, hue = 'Churn')


# In[80]:


y = df[['Churn']]
x = df.drop(['Churn', 'customerID'], axis = 1)


# In[81]:


x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 1029)


# In[82]:


mod = RandomForestClassifier().fit(x_train,y_train.values.ravel())


# In[83]:


mod.score(x_train,y_train)


# In[108]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 120, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[109]:


rf_mod = RandomForestClassifier()

RF_search = RandomizedSearchCV(estimator = rf_mod, param_distributions = random_grid, cv = 5, 
                               random_state=42)

RF_search.fit(x_train, y_train.values.ravel())


# In[110]:


RF_search.best_params_


# In[111]:


new_rf = RandomForestClassifier(n_estimators = 2000, min_samples_split = 5, min_samples_leaf = 2, max_features = 'auto',
                                max_depth = 48, bootstrap = True).fit(x_train, y_train.values.ravel())


# In[112]:


feature_imp = new_rf.feature_importances_


# In[113]:


indicies = np.argsort(feature_imp)[::-1]
features = x_train.columns

plt.figure(figsize = (30,4))
plt.bar(range(x_train.shape[1]), feature_imp[indicies], align = 'center')
plt.xticks(range(x_train.shape[1]), features[indicies])
plt.show()


# In[114]:


pd.DataFrame(RF_search.cv_results_)


# In[ ]:





# In[ ]:




