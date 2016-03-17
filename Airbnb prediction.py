
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[3]:

from xgboost.sklearn import XGBClassifier
np.random.seed(0)


# In[8]:

#Loading data
df_train = pd.read_csv('train_users.csv')
#df_test = pd.read_csv('test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
#id_test = df_test['id']
piv_train = df_train.shape[0]


# In[17]:

labels


# In[10]:

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)


# In[11]:

#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)


# In[74]:

df_all= pd.read_csv('data.csv')


# In[75]:

df_all.head()


# In[76]:

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)


# In[78]:

df_all.head()


# In[79]:

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)


# In[80]:

df_all.head()


# In[81]:

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>95), -1, av)


# In[18]:

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)


# In[4]:

df_all= pd.read_csv("Data_final.csv")


# In[88]:

#df_all['gender']=df_all['gender'].replace("-unknown-","unknown")


# In[5]:

#df_all= df_all.drop(['dac_month','dac_day','tfa_month','tfa_day'], axis=1)
df_all.head()


# In[12]:

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]


# In[20]:

print(X_test)


# In[53]:

from sklearn.ensemble import RandomForestClassifier
#from sklearn import cross_validation


# In[91]:

# Random Forests

random_forest = RandomForestClassifier(random_state=1, n_estimators=45, min_samples_split=3, min_samples_leaf=2)

random_forest.fit(X, y)
score=random_forest.score(X, y)

Y_pred = random_forest.predict(X_test)


# In[14]:

#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(X, y)
score = xgb.score(X,y)
y_pred = xgb.predict_proba(X_test)  



# In[15]:

print (score)


# In[21]:

# for Random forest


#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()



# In[24]:

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('AirBnb_xgb.csv',index=False)

