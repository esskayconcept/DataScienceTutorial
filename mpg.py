# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:47:23 2019

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
import warnings; warnings.simplefilter('ignore')

#  Loading Data
DataFile = "C:\\Users\\Somnath\\Dropbox\\Data Science\\Python Projects\\auto-mpg.csv"
raw_dataset = pandas.read_csv(DataFile)

#  Explore Data
print(raw_dataset.head(10))
raw_dataset.info()

#  Prepare Data
dataset = raw_dataset # Make a copy of the raw dataset; for safety

dataset['Make'], dataset['Model'] = dataset['car name'].str.split(' ', 1).str
print(dataset.head(10))
dataset.info()

#Handle Null
print("\n"+"*"*20 + " Null Values " +"*"*20 )
print(dataset.query("horsepower == '?'"))

# Cleaning Null Value
makeValuesWithNull = np.unique(dataset.query("horsepower == '?'")["Make"], return_counts=False)

for nc in makeValuesWithNull:
    ncMeanValue = np.mean(dataset.query("horsepower != '?' and Make =='" + nc + "'")['horsepower'])
    print(str(ncMeanValue))
    dataset.loc[(dataset["horsepower"] == '?') & (dataset["Make"] ==  nc),"horsepower"]  = ncMeanValue

print("\n"+"*"*20 + " Null Values " +"*"*20 )
print(dataset.query("horsepower == '?'"))

#  Separating the input fiels and the output columns
fields = dataset.iloc[:, :-3].values  
labels = dataset.iloc[:, 9].values  

# Exploring the output classes and their frequencies
labelValues = numpy.unique(labels, return_counts=True)
print(labelValues[0])
print(labelValues[1])

df = pd.DataFrame({'Make':labelValues[0],'Count':labelValues[1]}).sort_values(by='Count',ascending=True)

plt.figure(figsize=(12, 6))  
plt.barh(df['Make'], df['Count'])
plt.title('Frequency of different Car Make')

#  Separating the Data into training and testing
train_test_ratio =  0.8/0.2

# Find Best K:
noOfRows = dataset.shape[0]
minK = 5
maxK = int(noOfRows/2)
kValues = range(minK, maxK)
errors = [0.0] * len(kValues)

trials = 10

for j in range(0,trials):
    
    fields_train, fields_test, labels_train, labels_test = train_test_split(fields, labels, test_size = 1.0/(train_test_ratio + 1))
    
    # Normalizing Data
    scaler = StandardScaler()  
    scaler.fit(fields_train)
    
    fields_train = scaler.transform(fields_train)  
    fields_test = scaler.transform(fields_test) 
    
    for i in kValues:  
        
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(fields_train, labels_train)
        pred_i = knn.predict(fields_test)
        errors[i-minK] += (np.mean(pred_i != labels_test))

# errors = errors / trial

bestk = errors.index(np.min(errors[1:maxK])) + minK 

print("\nBest k : " + str(bestk))

plt.figure(figsize=(12, 6))  
plt.plot(kValues, errors, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Mean Error vs K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
