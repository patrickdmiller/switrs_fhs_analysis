#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Patrick Miller

experimenting with classifying serious injuries
credit for data set : Alex Gude https://alexgude.com/blog/switrs-sqlite-hosted-dataset/

'''


# In[21]:


import import_ipynb
import importlib


# In[45]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from time import perf_counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def to_onehot(col_name, df, prepend = '', drop = True):
    #change strings to categorical floats. 
    encoder = OneHotEncoder(sparse=True, categories='auto')
    transformer = encoder.fit_transform(df[[col_name]])
    names = encoder.categories_.copy()
    names = list(map(lambda x: prepend+'.'+x, names))
    df[names[0]] = transformer.toarray()
    if drop:
        df = df.drop(columns=[col_name])
    return df

from abc import ABC, abstractmethod

class DataClass(ABC):
    def _oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        return oversample.fit_resample(X_train, y_train)

    def _undersample(self, X_train, y_train):
        oversample = RandomUnderSampler(sampling_strategy='majority')
        return oversample.fit_resample(X_train, y_train)
    
    @abstractmethod
    def load(self, test_size = 0.3, undersample=False, oversample=False, verbose=False, scale=False, scale_type="minmax"):
        pass
    
    
class SwitrsData(DataClass):
    def __init__(self):
        self.df = None
    def test(self):
        to_onehot()
        
    def load(self, test_size = 0.3, undersample=False, oversample=False, verbose=False, scale=False, scale_type="minmax", force_reload = False, year='all'):
        if self.df is not None and not force_reload:
            print("loading from cache")
            df = self.df.copy()
        else:
            df = pd.read_csv('../../../data/switrs/out.csv')
            if year!='all':
                df = df[df['collision_date']==year]
                
            self.df = df.copy()
        print(df.columns)
        
        #do the oneshots
        onehots = ['weather', 'pcf_violation_category', 'lighting','road_condition', 'type_of_collision']
        for col in onehots:
            df = to_onehot(col, df, col, True)
        
        col_keep = [
            'state_highway_indicator', 'motorcycle_collision', 'bicycle_collision',
            'pedestrian_collision', 'alcohol_involved', 'severe_injury_count',
            'killed_victims', 'pedestrian_killed_count',
            'motorcyclist_killed_count', 'bicyclist_killed_count', 'weather._None_',
            'weather.clear', 'weather.cloudy', 'weather.fog', 'weather.other',
            'weather.raining', 'weather.snowing', 'weather.wind',
            'pcf_violation_category._None_',
            'pcf_violation_category.automobile right of way',
            'pcf_violation_category.brakes', 'pcf_violation_category.dui',
            'pcf_violation_category.fell asleep',
            'pcf_violation_category.following too closely',
            'pcf_violation_category.hazardous parking',
            'pcf_violation_category.impeding traffic',
            'pcf_violation_category.improper passing',
            'pcf_violation_category.improper turning',
            'pcf_violation_category.lights',
            'pcf_violation_category.other equipment',
            'pcf_violation_category.other hazardous violation',
            'pcf_violation_category.other improper driving',
            'pcf_violation_category.other than driver (or pedestrian)',
            'pcf_violation_category.pedestrian right of way',
            'pcf_violation_category.pedestrian violation',
            'pcf_violation_category.speeding',
            'pcf_violation_category.traffic signals and signs',
            'pcf_violation_category.unknown',
            'pcf_violation_category.unsafe lane change',
            'pcf_violation_category.unsafe starting or backing',
            'pcf_violation_category.wrong side of road',
            'lighting._None_', 'lighting.dark with no street lights',
            'lighting.dark with street lights',
            'lighting.dark with street lights not functioning', 'lighting.daylight',
            'lighting.dusk or dawn', 'road_condition._None_',
            'road_condition.construction', 'road_condition.flooded',
            'road_condition.holes', 'road_condition.loose material',
            'road_condition.normal', 'road_condition.obstruction',
            'road_condition.other', 'road_condition.reduced width',
            'type_of_collision._None_', 'type_of_collision.broadside',
            'type_of_collision.head-on', 'type_of_collision.hit object',
            'type_of_collision.other', 'type_of_collision.overturned',
            'type_of_collision.pedestrian', 'type_of_collision.rear end',
            'type_of_collision.sideswipe']
        
        df = df[col_keep]
        
        
        if scale:
            if scale_type=="minmax":
                scaler = MinMaxScaler(feature_range=(0,1)) 
                print("scaling data")
                df =  pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            else:
                ss = StandardScaler()
                scale_features = []
                df[scale_features] = ss.fit_transform(df[scale_features])
        y = df['severe_injury_count']
        X = df.drop('severe_injury_count', axis=1)
        if verbose:
            print("Loading", len(df), "rows")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1) # 70% training and 30% test
        if verbose:
            print("split: Train: ", len(X_train), len(y_train), "Test: ", len(X_test), len(y_test))
        if oversample:
            X_train, y_train = self._oversample(X_train, y_train)
        elif undersample:
            X_train, y_train = self._undersample(X_train, y_train)
        if (oversample or undersample) and verbose:
            print("after resample\nsplit: Train: ", len(X_train), len(y_train), "Test: ", len(X_train), len(y_train))
        return X_train, y_train, X_test, y_test




# In[46]:


# # df_test = pd.read_csv('../../../data/switrs/out.csv')
# switrsdata = SwitrsData()
# X_train, y_train, X_test, y_test = switrsdata.load(undersample=True, verbose=True, year=2018)


# In[49]:


# df_test = pd.read_csv('../../../data/switrs/out.csv')


# In[50]:


# df_test[df_test['collision_date']==2018]


# In[57]:


# print(len(df_test[(df_test['collision_date']==2018) & (df_test['severe_injury_count']==1)]), len(df_test[(df_test['collision_date']==2018) & (df_test['severe_injury_count']==0)]))

