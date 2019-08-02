# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:01:12 2019

@author: Jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import datetime

data = pd.read_csv('dataset.csv')

#%%

df = data[data.columns[1:]]
#df['WHOIS_UPDATED_DATE'] = pd.to_datetime(df['WHOIS_UPDATED_DATE']).apply(
#        lambda x: (datetime.datetime.today() - x).days)
df = df[[i for i in df.columns if i not in ('WHOIS_UPDATED_DATE','WHOIS_REGDATE')]]
cats = ['CHARSET','SERVER','WHOIS_COUNTRY','WHOIS_STATEPRO']
df = df[[i for i in df.columns if i not in cats]]

for c in cats[:-1]:
    d = pd.get_dummies(data[c])
    df = df.join(d,rsuffix=c+'_')
    
#%%
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

#%%

df.dropna(inplace=True)
X = df[[col for col in df.columns if col != 'Type']]
y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print('DECISION TREE:')
print(classification_report(y_test,y_pred))

tree = RandomForestClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print('FOREST :')
print(classification_report(y_test,y_pred))


#%%
fi = zip(X.columns,tree.feature_importances_)

X = X[[i[0] for i in fi if i[1] > 0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print('DECISION TREE LIMITED FEATURES:')
print(classification_report(y_test,y_pred))

print(cross_val_score(tree,X,y))

tree = RandomForestClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print('FOREST LIMITED FEATURES:')
print(classification_report(y_test,y_pred))

print(np.mean(cross_val_score(tree,X,y,cv=10)))

#%%