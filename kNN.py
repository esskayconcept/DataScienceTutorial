# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:00:47 2019

@author: Somnath
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix  
import seaborn as sn

#  Loading Data
DataFile = "C:\\Users\\Somnath\\Dropbox\\Data Science\\Python Projects\\iris.csv"
dataset = pandas.read_csv(DataFile)

#  Explore Data
print(dataset.head(10))
dataset.info()

#  Separating the fiels and the output columns
fields = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 4].values  

print(fields)
print(labels)

labelValues = numpy.unique(labels, return_counts=True)
print(labelValues[0])
print(labelValues[1])


#  Separating the Data into training and testing
fields_train, fields_test, labels_train, labels_test = train_test_split(fields, labels, test_size=0.20)

# Normalizing Data
scaler = StandardScaler()  
scaler.fit(fields_train)

fields_train = scaler.transform(fields_train)  
fields_test = scaler.transform(fields_test) 


# Get the Classifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric ='minkowski')  

# Let the model learn / train the model
classifier.fit(fields_train, labels_train)  

# Let the model predict
labels_pred = classifier.predict(fields_test)  

# Evaluate the Model
cm = confusion_matrix(labels_test, labels_pred)

print(cm)  
print(classification_report(labels_test, labels_pred))  


ax= plt.subplot()
sn.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

ax.set_xlabel('\nPrediction');ax.set_ylabel('Actual\n'); 
ax.set_title('Confusion Matrix\n'); 
ax.xaxis.set_ticklabels(labelValues); ax.yaxis.set_ticklabels(labelValues);


# Performance over value of k
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(fields_train, labels_train)
    pred_i = knn.predict(fields_test)
    error.append(np.mean(pred_i != labels_test))


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Mean Error vs K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 


# Performance over value of k
error = []

# Calculating error for p values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=5, p=i, metric = 'minkowski')
    knn.fit(fields_train, labels_train)
    pred_i = knn.predict(fields_test)
    error.append(np.mean(pred_i != labels_test))


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Mean Error vs P Value')  
plt.xlabel('P Value')  
plt.ylabel('Mean Error') 

# Calculating error for differnt distance metric 
distances = ["manhattan","euclidean","minkowski","chebyshev"]

err_dist = []

for d in distances:
    pval = 2
    if(d=="minkowski"):
        pval = 5
    knn = KNeighborsClassifier(n_neighbors=13, p=pval, metric = d)
    knn.fit(fields_train, labels_train)
    pred_i = knn.predict(fields_test)
    err_dist.append(np.mean(pred_i != labels_test))
    
    if(d=="minkowski"):
        distances[2] = "minkowski (" + str(pval) + ")"


print(err_dist)
print(distances)

plt.figure(figsize=(12, 6))  
plt.bar(distances, height = err_dist)
plt.title('Mean Error for differnt Distance metric')  