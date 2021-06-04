# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:45:13 2021

@author: Lenovo
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data_income = pd.read_csv("C:\\Users\\Lenovo\\Documents\\Documents\\python\\income.csv")

data = data_income.copy()

print (data.info())

data.isnull()


print (data.isnull().sum())

summary_num = data.describe()

print (summary_num)

summary_cate = data.describe(include='O')

print (summary_cate)

data['JobType'].value_counts()
data['occupation'].value_counts()
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

data = pd.read_csv("C:\\Users\\Lenovo\\Documents\\Documents\\python\\income.csv", na_values = [" ?"])

                                                                                             
data.isnull().sum()

missing = data[data.isnull().any(axis = 1 )]

data2 = data.dropna(axis = 0)
print (data2)
 
correlation = data2.corr()

print (correlation)


data2.columns

gender = pd.crosstab(index = data2["gender"],
                     columns = 'count',
                     normalize =True)
print (gender)

gender_salstat = pd.crosstab(index = data2['gender'],
                             columns = data2['SalStat'],
                             margins = True,
                             normalize = 'index')

print (gender_salsat)

salstat_jobtype = pd.crosstab(index = data2['JobType'],
                             columns = data2['SalStat'],
                             normalize = True)
                          
                           
print (salstat_jobtype)

SalStat = sns.countplot(data2['SalStat'])

sns.distplot(data2['age'],bins = 10,kde =False)

sns.boxplot('SalStat','age',data=data2)

data2.groupby('SalStat')['age'].median()

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000': 1})
print (data2['SalStat'])

new_data = pd.get_dummies(data,drop_first=True)

print (new_data)
columns_list = list(new_data.columns)
print (columns_list)

print (list(set(columns_list)))
print (set(['SalStat']))

features = list(set(columns_list)-set(['SalStat_ less than or equal to 50,000']))
print (features)


y = new_data['SalStat_ less than or equal to 50,000'].values
print (y)

x=new_data[features].values
print (x)

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

logistic = LogisticRegression()

logistic.fit(train_x,train_y)

logistic.coef_

logistic.intercept_
prediction = logistic.predict(test_x)

print (prediction)

confusion_matrix = confusion_matrix(test_y,prediction)
print (confusion_matrix)


accuracy_score = accuracy_score(test_y, prediction)
print (accuracy_score)

print('Misclassified samples: %d' % (test_y !=prediction).sum())


''' Run 2 - by removing insignificant columns

cols = ['gender', 'nativecountry', 'race','JobType']

new_data = data2.drop(cols, axis =1)


new_data = pd.get_dummies(new_data, drop_first = True)
print (new_data.columns)


columns_list = list(new_data.columns)

features = list(set(columns_list)-set(['SalStat_ less than or equal to 50,000']))
print (features)


y = new_data['SalStat'].values
print (y)

x = new_data[features].values
print (x)

train_x,test_x,train_y, test_y = train_test_split(x,y, test_size = 0.3, random_state = 0)

logistic = LogisticRegression()

logistic.fit(train_x,train_y)

prediction = logistic.predict(test_x)
accuracy_score = accuracy_score(test_y,prediction)

print (accuracy_score)

confusion_matrix = confusion_matrix(test_y,prediction)
print (confusion_matrix)'''