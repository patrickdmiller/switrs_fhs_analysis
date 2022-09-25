#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Patrick Miller

dataset:
https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data

'''
#install stuff if needed
get_ipython().system('pip install seaborn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install numpy')


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from time import perf_counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def _oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    return oversample.fit_resample(X_train, y_train)

def _undersample(X_train, y_train):
    oversample = RandomUnderSampler(sampling_strategy='majority')
    return oversample.fit_resample(X_train, y_train)

def load(test_size = 0.3, undersample=False, oversample=False, verbose=False, scale=False, scale_type="minmax"):
    df = pd.read_csv('../../../data/farmingham/data.csv')
    if scale:
        if scale_type=="minmax":
            scaler = MinMaxScaler(feature_range=(0,1)) 
            print("scaling data")
            df =  pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        else:
            ss = StandardScaler()
            print("scaling with standard scaler")
            scale_features = ["age","cigsPerDay","totChol","sysBP", "diaBP", "BMI", "heartRate", "glucose"]
            df[scale_features] = ss.fit_transform(df[scale_features])
    y = df['TenYearCHD']
    X = df.drop('TenYearCHD', axis=1)
    if verbose:
        print("Loading", len(df), "rows")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1) # 70% training and 30% test
    if verbose:
        print("split: Train: ", len(X_train), len(y_train), "Test: ", len(X_test), len(y_test))
    if oversample:
        X_train, y_train = _oversample(X_train, y_train)
    elif undersample:
        X_train, y_train = _undersample(X_train, y_train)
    if (oversample or undersample) and verbose:
        print("after resample\nsplit: Train: ", len(X_train), len(y_train), "Test: ", len(X_test), len(y_test))
    else:
        print("no resampling")
    return X_train, y_train, X_test, y_test
   
    

