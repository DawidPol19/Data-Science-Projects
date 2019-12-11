# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:56:56 2019
Melbourne Housig Prices
@author: Lofu
"""
#Import Packages
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#Reading Files
house_data = pd.read_csv("House Prices train.csv", index_col="Id")
X_test = pd.read_csv("House Prices test.csv", index_col="Id")

#Drop rows with missing targets, separating target from used predictors
house_data.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = house_data.SalePrice
house_data.drop(["SalePrice"], axis=1, inplace=True)

#Model test split
X_train, X_val, y_train, y_val = train_test_split(house_data, y, train_size=0.8, random_state=1)

#Select columns with low unique categories
low_cardinality_cols = [cname for cname in X_train if X_train[cname].nunique() < 10 and X_train[cname].dtype == "object"]

#Select numeric columns
numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ["int64", "float64"]]

#Keep Selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train[my_cols].copy()
X_valid = X_val[my_cols].copy()
X_test = X_test[my_cols].copy()

#One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join="left", axis=1)
X_train, X_test = X_train.align(X_test, join="left", axis=1)

#Pandas Exploratory Stats
print(house_data.head(10))
print(house_data.describe())
print(house_data.shape)
print(house_data.columns)

#Model Creation Testing
res = []
counters = 720
model_1 = XGBRegressor(n_estimators=counters)
model_1.fit(X_train, y_train)
preds_1 = model_1.predict(X_valid)
mae_1 = mean_absolute_error(preds_1, y_val)
res.append(mae_1)

#Test data modeling
f = res.index(min(res))
model_1.fit(X_train, y_train)
preds = model_1.predict(X_test)

#Data Output
test_cols = pd.read_csv(r"C:\Users\Lofu\Desktop\Python Machine Learning\Data Sets\House Prices test.csv")
output = pd.DataFrame({"Id": test_cols.Id, "SalePrice": preds})
output.to_csv("House submission.csv", index=False)

