#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from random import randrange, uniform


# In[2]:


#set the working directory
os.chdir("E:/Data Science_ Edwisor/Cab Fare Prediction")


# In[3]:


#checking the working directory
os.getcwd()


# In[4]:


#loading the train data
train_cabdata = (pd.read_csv("train_cab.csv",header=0)).drop(columns="pickup_datetime")


# In[5]:


train_cabdata.shape


# In[7]:


#removing the rows which have same pickup and drop location
train_cabdata = train_cabdata[np.logical_and(train_cabdata['pickup_longitude'] != train_cabdata['dropoff_longitude'],
                                     train_cabdata['pickup_latitude'] != train_cabdata['dropoff_latitude'])]


# In[8]:


#we will now replace 0 with NA in the variables and convert the data if required anywhere for further operations

train_cabdata['fare_amount']= train_cabdata['fare_amount'].apply(pd.to_numeric, errors='coerce')
train_cabdata['fare_amount']= train_cabdata['fare_amount'].replace({0:np.nan})
train_cabdata['passenger_count']=train_cabdata['passenger_count'].fillna(0)
train_cabdata['passenger_count']= train_cabdata['passenger_count'].astype(int)
train_cabdata['passenger_count']=train_cabdata['passenger_count'].replace({0: np.nan})
train_cabdata['pickup_longitude']= train_cabdata['pickup_longitude'].replace({0:np.nan})
train_cabdata['pickup_latitude']= train_cabdata['pickup_latitude'].replace({0:np.nan})
train_cabdata['dropoff_longitude']= train_cabdata['dropoff_longitude'].replace({0:np.nan})
train_cabdata['dropoff_latitude']= train_cabdata['dropoff_latitude'].replace({0:np.nan})


# In[9]:


#checking the dimension of our dataset - train_cabdata
train_cabdata.shape


# In[10]:


#viewing the top 10 data in the train_cabdata
train_cabdata.head(10)


# Missing Value Analysis

# In[11]:


#we will calculate the missing values

missing_val = pd.DataFrame(train_cabdata.isnull().sum())
print(missing_val)


# In[12]:


#Reset  the index
missing_val = missing_val.reset_index()

#Renaming the variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'count'})

#we will Calculate percentage
missing_val['Missing_percentage'] = (missing_val['count']/len(train_cabdata)*100)

#sort in descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
print(missing_val)


# Imputation of the Missing Values

# In[13]:


#Create missing value, in order to identify the method which is better for imputation

train_cabdata["pickup_longitude"].loc[70]


# In[14]:


#imputation method 
#Actual value = -73.99578100000001
#Mean = -73.91159336554888
#Median = -73.9820605
#KNN = -73.890529


# In[15]:


#replacing the value with nan
train_cabdata["pickup_longitude"].loc[70] = np.nan


# In[17]:


#checking the missing value created
train_cabdata["pickup_longitude"].loc[70]


# In[20]:


#mean imputation
train_cabdata['pickup_longitude'] = train_cabdata['pickup_longitude'].fillna(train_cabdata['pickup_longitude'].mean())

train_cabdata["pickup_longitude"].loc[70]


# In[19]:


#median imputation
#train_cabdata['pickup_longitude'] = train_cabdata['pickup_longitude'].fillna(train_cabdata['pickup_longitude'].median())

#train_cabdata["pickup_longitude"].loc[70]


# In[21]:


#imputation using KNN
#train_cabdata = pd.DataFrame(KNN(k = 1).fit_transform(train_cabdata), columns = train_cabdata.columns)

#train_cabdata["pickup_longitude"].loc[70]


# In[22]:


#we find that the mean value is closest to the actual value , hence we will proceed with mean imputation method
train_cabdata['fare_amount'] = train_cabdata['fare_amount'].fillna(train_cabdata['fare_amount'].mean())
train_cabdata['pickup_longitude']= train_cabdata['pickup_longitude'].fillna(train_cabdata['pickup_longitude'].mean())
train_cabdata['pickup_latitude']= train_cabdata['pickup_latitude'].fillna(train_cabdata['pickup_latitude'].mean())
train_cabdata['dropoff_longitude']= train_cabdata['dropoff_longitude'].fillna(train_cabdata['dropoff_longitude'].mean())
train_cabdata['dropoff_latitude']= train_cabdata['dropoff_latitude'].fillna(train_cabdata['dropoff_latitude'].mean())


#for categorical variable will use mode imputaton method
train_cabdata['passenger_count'] = train_cabdata['passenger_count'].fillna(int(train_cabdata['passenger_count'].mode()))


# In[23]:


#removing rows that have na as their presence will hamper the data
train_cabdata = train_cabdata.dropna()


# In[24]:


#converting into proper data type
convert_traindata = {'fare_amount':'float','passenger_count':'int'}
train_cabdata = train_cabdata.astype(convert_traindata)


# In[25]:


#checking the dimensions of the train data
train_cabdata.shape


# Outlier Analysis

# In[27]:


#saving copy of the data
df = train_cabdata.copy()
train_cabdata = train_cabdata.copy()


# In[28]:


#fare_amount having irregular data are converted to NA
train_cabdata.loc[train_cabdata['fare_amount']<0,'fare_amount']=np.nan
train_cabdata.loc[train_cabdata['fare_amount']>30,'fare_amount']=np.nan
train_cabdata = train_cabdata.dropna()


# In[29]:


#passenger_count which are irregular i.e, greaater than 8 are converted to NA
train_cabdata.loc[train_cabdata['passenger_count']>8,'passenger_count']=np.nan


# In[30]:


#saving numeric names
cnames = [ 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
for i in cnames:
    #Detect and replace with NA
    #Extract quartiles
    q75, q25 = np.percentile(train_cabdata[i], [75 ,25])

    #Calculate IQR
    iqr = q75 - q25
   
    #Calculate inner/lower and outer/upper fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)

    #Replace with NA
    train_cabdata.loc[train_cabdata[i] < minimum,i] = np.nan
    train_cabdata.loc[train_cabdata[i] > maximum,i] = np.nan

    #Calculate missing value
    missing_val = pd.DataFrame(train_cabdata.isnull().sum())


# In[32]:


#Since mean is the best imputation method, we impute the outlier values with mean

train_cabdata['pickup_longitude'] = train_cabdata['pickup_longitude'].fillna(train_cabdata['pickup_longitude'].mean())


# In[34]:


train_cabdata['pickup_latitude'] = train_cabdata['pickup_latitude'].fillna(train_cabdata['pickup_latitude'].mean())


# In[35]:


train_cabdata['dropoff_longitude']=train_cabdata['dropoff_longitude'].fillna(train_cabdata['dropoff_longitude'].mean())
train_cabdata['dropoff_latitude']=train_cabdata['dropoff_latitude'].fillna(train_cabdata['dropoff_latitude'].mean())


# In[36]:


#imputed with mode for categorical variables
train_cabdata['passenger_count'] = train_cabdata['passenger_count'].fillna(int(train_cabdata['passenger_count'].mode()))


# In[37]:


#converting data type of passenger_count
train_cabdata['passenger_count']=train_cabdata['passenger_count'].astype('int')
train_cabdata['passenger_count']=train_cabdata['passenger_count'].astype('category')


# Feature Selection

# In[38]:


#haversine function

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
   
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


# In[40]:


#defining new variable
train_cabdata['dist'] =haversine( train_cabdata['pickup_latitude'], train_cabdata['pickup_longitude'],
                train_cabdata['dropoff_latitude'], train_cabdata['dropoff_longitude'])


# In[41]:


#Correlation analysis
#Correlation plot
cab_numericdata=['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude', 'dist']
train_cabdata_corr = train_cabdata.loc[:,cab_numericdata]


# In[42]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr_matrix = train_cabdata_corr.corr()
print(corr_matrix)

#Plotted using seaborn library
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[43]:


#we will now eliminate the data having same pickup and drop locations
train_cabdata = train_cabdata[np.logical_and(train_cabdata['pickup_longitude'] != train_cabdata['dropoff_longitude'],
                                            train_cabdata['pickup_latitude'] !=train_cabdata['dropoff_latitude'])]


# Model Development

# Decision Tree

# In[44]:


#loading the required libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[45]:


#dividing the data into train and test
train, test = train_test_split(train_cabdata, test_size=0.2)


# In[47]:


#decision tree for regression
#max node=2
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:, 1:7], train.iloc[:,0])


# In[48]:


fit_DT


# In[49]:


#applying model on the test data
predictions_DT = fit_DT.predict(test.iloc[:,1:7])


# In[50]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[51]:


MAPE(test.iloc[:,0],predictions_DT)


# In[52]:


#error 28.159791420474832
#accuracy 71.84


# Random Forest

# In[53]:


#importing required libraries
from sklearn.ensemble import RandomForestRegressor


# In[54]:


RF_model = RandomForestRegressor(n_estimators = 10).fit(train.iloc[:, 1:7], train.iloc[:,0])


# In[55]:


RF_model


# In[56]:


RF_Predictions = RF_model.predict(test.iloc[:, 1:7])


# In[57]:


MAPE(test.iloc[:,0],RF_Predictions)


# In[58]:


#error 23.65777233717846
#accuracy 76.34


# Linear Regression

# In[59]:


#putting all the data values in one array
values=['fare_amount', 'pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'dist']


# In[60]:


lin_Data = train_cabdata[values]


# In[61]:


#This function is developed to get columns for specific passenger count 

cat_names = ['passenger_count'] 
for i in cat_names:
    temp = pd.get_dummies(train_cabdata[i], prefix= i)
    lin_Data = lin_Data.join(temp)


# In[62]:


#checking the dimensions
lin_Data.shape


# In[63]:


#viewing the data
lin_Data.head()


# In[64]:



#now we will split the above created data set with passenger count dummies
train, test = train_test_split(lin_Data, test_size=0.2)


# In[65]:


#importing required libraries for LR
import statsmodels.api as sm


# In[66]:


#train the model using the training sets
model = sm.OLS(train.iloc[:, 0].astype(float), train.iloc[:, 1:12].astype(float)).fit()


# In[67]:


#print the summary
model.summary()


# In[68]:


# make the predictions
predictions_LR = model.predict(test.iloc[:,1:12])


# In[69]:


#calculate mape
MAPE(test.iloc[:,0],predictions_LR)


# In[70]:


#error 44.949335728466224
#accuracy 55.05


# KNN Imputation

# In[71]:


#KNN implementation
from sklearn.neighbors import KNeighborsRegressor

KNN_model = KNeighborsRegressor(n_neighbors = 1).fit(train.iloc[: , 1:7], train.iloc[:, 0])


# In[73]:


#predict test cases
KNN_Predictions = KNN_model.predict(test.iloc[: , 1:7])


# In[74]:


#calculate mape
MAPE(test.iloc[:,0],KNN_Predictions)


# In[75]:


#error 42.42299709564156
#accuracy 57.57


# Prediction on original test data

# In[76]:


original_test=(pd.read_csv('test.csv', header = 0 )).drop(columns="pickup_datetime")


# In[77]:


#create Dist variable
original_test['dist'] = haversine( original_test['pickup_latitude'], original_test['pickup_longitude'],
                 original_test['dropoff_latitude'], original_test['dropoff_longitude'])

original_test['fare_amount']=0
original_test['passenger_count']=original_test['passenger_count'].astype('category')


# In[78]:


# Build model on the entire the entire data
RF_model = RandomForestRegressor(n_estimators = 10).fit(train_cabdata.iloc[:, 1:7], train_cabdata.iloc[:,0])

#predict value
original_test['fare_amount'] = RF_model.predict(original_test.iloc[:, 0:6])


# In[81]:


#viewing the predicted data
original_test.head(10)


# In[80]:


#saving the output of predicted value in csv
original_test.to_csv("predicted_values.csv",index=False)


# In[ ]:




