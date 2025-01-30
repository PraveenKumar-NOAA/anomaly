#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 13:59:45 2023

@author: Praveen Singh
"""

import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
from pyGSI.diags import Conventional
from pathlib import Path

# 1. Read Dataset and convert to csv
gsifile = "./data_2021080100/gdas.t00z.adpupa_prepbufr.tm00.nc"
diag = Conventional(gsifile)
df = diag.get_data()
df.to_csv('/work2/noaa/da/pkumar/Project/temp_gsi.csv',index=True, header=True)
print('Head of the Dataset')
print(df.head())

# 2. Create a new subset of few variables
#newdf = df[['Station_ID', 'latitude', 'longitude', 'Pressure', 'observation']]
newdf = df[['latitude', 'longitude', 'time', 'observation']]

# 3. Describe the data
print('Head of the sub Dataset')
print(newdf.head())
print('Tail of the sub Dataset')
print(newdf.tail())
print('Describe the sub Dataset')
print(newdf.describe())
print('Histograms of the sub Dataset')
#newdf.hist()
#plt.show()

# 3.1 Change time to datetime format
#newdf['time'] = pd.to_datetime(newdf['time'], format='%Y-%m')
#newdf['time'] = newdf['time'], format='%Y-%m-%d %H:%M:%S')
#print(newdf.head())

# 4. Correlation
print(newdf.corr())

# 5. Check missing values
newdf.isnull().sum()
missing_count = newdf.isnull().sum() # the count of missing values
value_count = newdf.isnull().count() # the count of all values
missing_percentage = round(missing_count / value_count * 100, 1)
missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage})
print(missing_df)

# 6. Create the time-series plot
plt.plot(newdf['time'], newdf['observation'], linestyle='dotted')
# Add title and axis labels
plt.title('Time Series Plot')
plt.xlabel('Time')
plt.ylabel('Observations')
plt.xticks(rotation=45)
plt.xlim(2.9, 3)
#plt.show()
#exit()

# 7. ML/ DBSCAN: Unsupervised learning
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

print(newdf.head())
print(newdf.columns)
print(f"number of rows: {len(newdf)}")

# Scale and normalize
scaler = StandardScaler()

newdf_s = scaler.fit_transform(newdf)
newdf_norm = pd.DataFrame(normalize(newdf_s))
print(newdf_norm.head())


pca = PCA(n_components = 2)
newdf_principal = pca.fit_transform(newdf_norm)
newdf_principal = pd.DataFrame(newdf_principal)
newdf_principal.columns = ['P1', 'P2']

db_model = DBSCAN(eps = 0.05, min_samples = 10).fit(newdf_principal)
labels = db_model.labels_

np.unique(labels)

np.histogram(labels, bins=len(np.unique(labels)))
print(np.histogram(labels, bins=len(np.unique(labels))))
#plt.hist(labels, bins=len(np.unique(labels)), log=True)
#plt.show()

n_clusters = len(np.unique(labels))-1
anomaly = list(labels).count(-1)
print(f'Clusters: {n_clusters}')
print(f'Abnormal points: {anomaly}')

import seaborn as sns
plt.figure()
sns.scatterplot(
    x="P1", y="P2",
    palette=sns.color_palette("hls", 10),
    data=newdf_principal,
    legend="full",
    alpha=0.3
)
#plt.show()


# 8. Split the dataset into training and testing parts using sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
print(sklearn.__version__)
X = newdf.iloc[:,:-1]
print(X)
y = newdf.iloc[:, 3]
print(y)

# https://www.geeksforgeeks.org/how-to-split-a-dataset-into-train-and-test-sets-using-python/
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.7, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 8.1 Scale the training data towards unit variance (mean=0  variance=1)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# 8.2 Implement logistic regression model, train, and test: Supervised learning
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(max_iter = 1000)
y_train = y_train.astype('int')
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
# print the train test split and the accuracy of the test
from sklearn import metrics
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

reg.score(X_test, y_pred)
print("Accuracy:",reg.score(X_test, y_pred))
exit()
