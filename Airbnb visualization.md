

```python
# Airbnb data exploration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Draw inline
%matplotlib inline
```


```python
# Set figure aesthetics
import seaborn as sns
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)
```

    C:\Anaconda3\lib\site-packages\matplotlib\__init__.py:872: UserWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      warnings.warn(self.msg_depr % (key, alt_key))
    


```python
# Load the data into DataFrames
train_users = pd.read_csv('train_users.csv')
test_users = pd.read_csv('test_users.csv')
```


```python
# Merge train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Remove ID's since now we are not interested in making predictions
users.drop('id',axis=1, inplace=True)

users.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>affiliate_channel</th>
      <th>affiliate_provider</th>
      <th>age</th>
      <th>country_destination</th>
      <th>date_account_created</th>
      <th>date_first_booking</th>
      <th>first_affiliate_tracked</th>
      <th>first_browser</th>
      <th>first_device_type</th>
      <th>gender</th>
      <th>language</th>
      <th>signup_app</th>
      <th>signup_flow</th>
      <th>signup_method</th>
      <th>timestamp_first_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>direct</td>
      <td>direct</td>
      <td>NaN</td>
      <td>NDF</td>
      <td>2010-06-28</td>
      <td>NaN</td>
      <td>untracked</td>
      <td>Chrome</td>
      <td>Mac Desktop</td>
      <td>-unknown-</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>facebook</td>
      <td>20090319043255</td>
    </tr>
    <tr>
      <th>1</th>
      <td>seo</td>
      <td>google</td>
      <td>38</td>
      <td>NDF</td>
      <td>2011-05-25</td>
      <td>NaN</td>
      <td>untracked</td>
      <td>Chrome</td>
      <td>Mac Desktop</td>
      <td>MALE</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>facebook</td>
      <td>20090523174809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>direct</td>
      <td>direct</td>
      <td>56</td>
      <td>US</td>
      <td>2010-09-28</td>
      <td>2010-08-02</td>
      <td>untracked</td>
      <td>IE</td>
      <td>Windows Desktop</td>
      <td>FEMALE</td>
      <td>en</td>
      <td>Web</td>
      <td>3</td>
      <td>basic</td>
      <td>20090609231247</td>
    </tr>
    <tr>
      <th>3</th>
      <td>direct</td>
      <td>direct</td>
      <td>42</td>
      <td>other</td>
      <td>2011-12-05</td>
      <td>2012-09-08</td>
      <td>untracked</td>
      <td>Firefox</td>
      <td>Mac Desktop</td>
      <td>FEMALE</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>facebook</td>
      <td>20091031060129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>direct</td>
      <td>direct</td>
      <td>41</td>
      <td>US</td>
      <td>2010-09-14</td>
      <td>2010-02-18</td>
      <td>untracked</td>
      <td>Chrome</td>
      <td>Mac Desktop</td>
      <td>-unknown-</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>basic</td>
      <td>20091208061105</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Missing Data
users.gender.replace('-unknown-', np.nan, inplace=True)

```


```python
users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')
```




    age                        42.412365
    date_first_booking         67.733998
    first_affiliate_tracked     2.208335
    gender                     46.990169
    dtype: float64




```python
users.age.describe()
print(sum(users.age > 122))
print(sum(users.age < 18))
```

    830
    188
    


```python
users[users.age < 14]["age"].describe()
```




    count    59.000000
    mean      4.322034
    std       1.331847
    min       1.000000
    25%       5.000000
    50%       5.000000
    75%       5.000000
    max       5.000000
    Name: age, dtype: float64




```python
users.loc[users.age > 95, 'age'] = np.nan
users.loc[users.age < 14, 'age'] = np.nan
```


```python
categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')
```


```python
# formatting date
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
```


```python
# Graph by Gender
users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Gender')
sns.despine()
```


![png](output_11_0.png)



```python
women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100

# Bar width
width = 0.4

male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()
```


![png](output_12_0.png)



```python
destination_percentage = users.country_destination.value_counts() / users.shape[0] * 100
destination_percentage.plot(kind='bar',color='#FD5C64', rot=0)
# Using seaborn can also be plotted
# sns.countplot(x="country_destination", data=users, order=list(users.country_destination.value_counts().keys()))
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
#The first thing we can see that if there is a reservation, it's likely to be inside the US. 
#But there is a 45% of people that never did a reservation.
```


![png](output_13_0.png)



```python
sns.distplot(users.age.dropna(), color='#FD5C64')
plt.xlabel('Age')
sns.despine()
# the common age to travel is between 25 and 40.
```


![png](output_14_0.png)



```python
# Let's see if, for example, older people travel in a different way. 
#Let's pick an arbitrary age to split into two groups. Maybe 45?
age = 45

younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())

younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100

younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Youngers', rot=0)
older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Olders', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()
#We can see that the young people tends to stay in the US, and the older people choose to travel outside the country
```


![png](output_15_0.png)



```python

print((sum(users.language == 'en') / users.shape[0])*100)
# With the 96% of users using English as their language, it is understandable that a lot of people stay in the US.
```

    96.3675888324
    


```python
sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)
users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fef0de400>




![png](output_17_1.png)



```python
users.date_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ff3c68f98>




![png](output_18_1.png)



```python
users_2013 = users[users['date_first_active'] > pd.to_datetime(20130101, format='%Y%m%d')]
users_2013 = users_2013[users_2013['date_first_active'] < pd.to_datetime(20140101, format='%Y%m%d')]
users_2013.date_first_active.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
plt.show()

```


![png](output_19_0.png)



```python
weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
weekdays = pd.Series(weekdays)
sns.barplot(x = weekdays.value_counts().index, y=weekdays.value_counts().values, order=range(0,7))
plt.xlabel('Week Day')
sns.despine()
```

    C:\Anaconda3\lib\site-packages\matplotlib\__init__.py:892: UserWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      warnings.warn(self.msg_depr % (key, alt_key))
    


![png](output_20_1.png)



```python
date = pd.to_datetime(20140101, format='%Y%m%d')

before = sum(users.loc[users['date_first_active'] < date, 'country_destination'].value_counts())
after = sum(users.loc[users['date_first_active'] > date, 'country_destination'].value_counts())
before_destinations = users.loc[users['date_first_active'] < date, 
                                'country_destination'].value_counts() / before * 100
after_destinations = users.loc[users['date_first_active'] > date, 
                               'country_destination'].value_counts() / after * 100
before_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Before 2014', rot=0)
after_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='After 2014', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()
```


![png](output_21_0.png)

