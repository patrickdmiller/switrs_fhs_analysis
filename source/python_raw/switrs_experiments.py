#!/usr/bin/env python
# coding: utf-8

# In[48]:


import import_ipynb
import importlib
import switrs_data
from switrs_data import SwitrsData
importlib.reload(switrs_data)
switrsdata = SwitrsData()


# In[44]:


get_ipython().system('pip install scikit-learn-intelex')


# In[49]:



from sklearnex import patch_sklearn 
patch_sklearn()
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from time import perf_counter
from sklearn.model_selection import StratifiedKFold, learning_curve
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


'''
      _           _     _               _______                 
     | |         (_)   (_)             |__   __|                
   __| | ___  ___ _ ___ _  ___  _ __      | |_ __ ___  ___  ___ 
  / _` |/ _ \/ __| / __| |/ _ \| '_ \     | | '__/ _ \/ _ \/ __|
 | (_| |  __/ (__| \__ \ | (_) | | | |    | | | |  __/  __/\__ \
  \__,_|\___|\___|_|___/_|\___/|_| |_|    |_|_|  \___|\___||___/
                                                                
'''
from sklearn.tree import DecisionTreeClassifier


# In[51]:


def best_score(scores, SCORE_TYPE):
    best = {key: {"label":"", "score":0} for key in SCORE_TYPE}
    
    
    for i in range(len(scores)):
        for s in SCORE_TYPE:
            if scores[i]['scores'][s] > best[s]["score"]:
                best[s]["label"] = scores[i]['label']
                best[s]["score"] = scores[i]['scores'][s]
    return best
        


# In[52]:


X_train, y_train, X_test, y_test = switrsdata.load(undersample=True, verbose=True, year=2018)


# In[54]:


#first create decision tree with varying depth
SCORE_TYPE = ["accuracy", "f1",]
scores = []
for i in range(1,40):
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth = i,
        criterion="entropy"
    )
    scores.append({
        "label":f'max_depth_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(clf, X_train, y_train,scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t


# In[55]:


knn_k_scores= scores
print(best_score(knn_k_scores, SCORE_TYPE))
knn_k_scores


# In[58]:


#find the best feature count
scores = []
for i in range(1,20):
    clf = DecisionTreeClassifier(
        random_state=42,
        criterion="entropy",
        max_features=i
    )
    scores.append({
        "label":f'max_features_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(clf, X_train, y_train,scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t


# In[59]:


#print the results (used for graphing as well)
min_samples_scores = scores
print(best_score(min_samples_scores, SCORE_TYPE))
min_samples_scores


# In[63]:


#print the results (used for graphing as well)
training_scores = scores
training_scores
estimator = clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=12,
        criterion="entropy",
        max_features=16
    )
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X_train, y_train, cv=6,return_times=True, n_jobs=6)


# In[64]:


estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")
plt.legend(loc="best")


plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
print("f1:",metrics.f1_score(y_test, y_pred))
print("acc", metrics.accuracy_score(y_test, y_pred))
plt.xlabel("Num Samples")
plt.ylabel("Score")
plt.legend(loc="best")


# In[78]:


#best is depth 18 and min sample split of 13
#test set on best
t = perf_counter()
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=18,
    criterion="entropy",
    max_features=13
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
print(perf_counter() - t)


# In[65]:


'''
  _                     _   
 | |                   | |  
 | |__   ___   ___  ___| |_ 
 | '_ \ / _ \ / _ \/ __| __|
 | |_) | (_) | (_) \__ \ |_ 
 |_.__/ \___/ \___/|___/\__|
                            
                            
'''
from sklearn.ensemble import AdaBoostClassifier


# In[66]:


scores = []
SCORE_TYPE = ["accuracy"]
n_i=[50,500,1000]
for i in n_i:
    print(i)
    base = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    model = AdaBoostClassifier(base_estimator=base,  n_estimators=(i))
    scores.append({
        "label":f'n_estimators_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(model, X_train, y_train, scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t


# In[67]:


#print the results (used for graphing as well)
n_estimators_score = scores
print(best_score(n_estimators_score, SCORE_TYPE))
n_estimators_score


# In[68]:


scores = []
SCORE_TYPE = ["accuracy"]
d = [1,3,5,7,9]
for i in d:
    print(i)
    base = DecisionTreeClassifier(max_depth=i, criterion="entropy")
    model = AdaBoostClassifier(base_estimator=base,  n_estimators=500)
    scores.append({
        "label":f'n_estimators_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(model, X_train, y_train, scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t


# In[69]:


max_depth_boost = scores
print(best_score(max_depth_boost, SCORE_TYPE))
max_depth_boost


# In[128]:


base = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base, n_estimators=500)
# model.fit(X_train, y_train)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=6,return_times=True, n_jobs=6)



plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")


model.fit(X_train, y_train)
t = perf_counter()
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Precision:",metrics.recall_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")

plt.legend(loc="best")
print(perf_counter()-t)


# In[74]:


'''
  _____   ___ __ ___  
 / __\ \ / / '_ ` _ \ 
 \__ \\ V /| | | | | |
 |___/ \_/ |_| |_| |_|
                      
                      
'''
from sklearn.svm import SVC
X_train, y_train, X_test, y_test = switrsdata.load(undersample=True, verbose=True, year=2018, test_size=0.5)


# In[75]:


scores = []
print(len(X_train))
SCORE_TYPE = ["accuracy"]
KERNELS = ['linear','poly', 'rbf','sigmoid']

for k in KERNELS:
    print(k)
    svm = SVC(kernel=k)

    scores.append({
        "label":f'kernel_{k}_degree_3',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(svm, X_train, y_train, scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t


# In[76]:


svm_scores= scores
print(best_score(svm_scores, SCORE_TYPE))
svm_scores


# In[77]:


scores = []
SCORE_TYPE = ["accuracy"d]
KERNELS = ['rbf']
divs = [ 1,10,100,1000,10000]
for i in divs:
    for k in KERNELS:
        svm = SVC(kernel=k, gamma=1/i)
  
        scores.append({
            "label":f'kernel_{k}_gamma_{1/i}',
            "scores":{},
            "times":{}
        })
        for s in SCORE_TYPE:
            t = perf_counter()
            scores[-1]["scores"][s] = sum(cross_val_score(svm, X_train, y_train, scoring=s, cv=6, n_jobs=6))/6
            scores[-1]["times"][s] = perf_counter() - t


# In[78]:


svm_scores= scores
print(best_score(svm_scores, SCORE_TYPE))
svm_scores


# In[79]:


svm = SVC(kernel="rbf", gamma = 0.1)
svm.fit(X_train, y_train)
print("done fitting")
t = perf_counter()
y_pred = svm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
print(perf_counter() - t)


# In[80]:


model = SVC(kernel="rbf", gamma = 0.1)
# model.fit(X_train, y_train)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=6,return_times=True, n_jobs=6)

plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")


# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")


plt.legend(loc="best")
# print(perf_counter()-t)


# In[83]:


'''
  _                
 | |               
 | | ___ __  _ __  
 | |/ / '_ \| '_ \ 
 |   <| | | | | | |
 |_|\_\_| |_|_| |_|
                   
                   
'''

from sklearn.neighbors import KNeighborsClassifier
# X_train, y_train, X_test, y_test = switrsdata.load(undersample=True, verbose=True, test_size=0.75)
X_train, y_train, X_test, y_test = switrsdata.load(undersample=True, verbose=True, year=2018)


# In[84]:


scores = []
SCORE_TYPE = ["accuracy"]
for i in range(1, 16, 1):
    print(i)
    knn = KNeighborsClassifier(n_neighbors=i)

    scores.append({
        "label":f'knn_k_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(knn, X_train, y_train, scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t


# In[85]:


knn_k_scores= scores
print(best_score(knn_k_scores, SCORE_TYPE))
knn_k_scores


# In[86]:


scores = []
SCORE_TYPE = ["accuracy"]
algos= ['auto','ball_tree','kd_tree','brute']
weights = ["uniform","distance"]

for l in range(5,100,10):
    print(l)
    knn = KNeighborsClassifier(n_neighbors=15, leaf_size=l)

    scores.append({
        "label":f'knn_l_{l}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(knn, X_train, y_train, scoring=s, cv=6, n_jobs=6))/6
        scores[-1]["times"][s] = perf_counter() - t
knn_k_scores = scores
print(best_score(knn_k_scores, SCORE_TYPE))


# In[87]:


knn_k_scores= scores
print(best_score(knn_k_scores, SCORE_TYPE))
knn_k_scores


# In[129]:


model = KNeighborsClassifier(n_neighbors=15)
# model.fit(X_train, y_train)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=6,return_times=True, n_jobs=6)

plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")


model.fit(X_train, y_train)
t = perf_counter()
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")


plt.legend(loc="best")
print(perf_counter()-t)


# In[98]:


'''
                             _            _                      _    
                            | |          | |                    | |   
  _ __   ___ _   _ _ __ __ _| |_ __   ___| |___      _____  _ __| | __
 | '_ \ / _ \ | | | '__/ _` | | '_ \ / _ \ __\ \ /\ / / _ \| '__| |/ /
 | | | |  __/ |_| | | | (_| | | | | |  __/ |_ \ V  V / (_) | |  |   < 
 |_| |_|\___|\__,_|_|  \__,_|_|_| |_|\___|\__| \_/\_/ \___/|_|  |_|\_\
                                                                      
                                                                      
'''
X_train, y_train, X_test, y_test = switrsdata.load(undersample=True, verbose=True, year=2018, test_size=0.80)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train_validate_fold(model, _X_train, _y_train, _X_val, _y_val, epochs=200):
    model.get_weights()
    print("training for ", epochs, "epochs")
    model.fit(_X_train, _y_train, epochs=epochs, verbose = 0)
    scores = model.evaluate(_X_val, _y_val, verbose=0)
    return scores


# In[99]:


def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['accuracy'][-1]
    plt.title('Accuracy vs Loss (with validation)')


# In[100]:


scores = []
nodes_to_try = [16,32,64, 128]
for n in nodes_to_try:
    print( " -- hidden", n ,"--")
    model = Sequential()
    model.add(Dense(n, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    kfold.get_n_splits(X_train, y_train)
    average_scores = [0,0]
    i = 0
    runtime= perf_counter()
    for t, v in kfold.split(X_train, y_train):
        _scores = train_validate_fold(model, X_train.loc[t], y_train.loc[t], X_train.loc[v], y_train.loc[v], epochs=300)
        average_scores[0]+=_scores[0]
        average_scores[1]+=_scores[1]
        i+=1
    scores.append({
            "label":f'nn_n_{n}',
            "scores":[average_scores[0]/i, average_scores[1]/i],
            "times":(perf_counter() - runtime)/2
        })
    print("time total: ", perf_counter() - runtime)


# In[101]:


scores


# In[102]:


scores = []
nodes_to_try = [16,32,64, 128]
learning_rates = [.1, .01, .001, .0001]

for lr in learning_rates:

    print( " -- lr", lr ,"--")
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    kfold.get_n_splits(X_train, y_train)
    average_scores = [0,0]
    i = 0
    runtime= perf_counter()
    for t, v in kfold.split(X_train, y_train):
        _scores = train_validate_fold(model, X_train.loc[t], y_train.loc[t], X_train.loc[v], y_train.loc[v], epochs=300)
        average_scores[0]+=_scores[0]
        average_scores[1]+=_scores[1]
        i+=1
    scores.append({
            "label":f'nn_lr_{lr}',
            "scores":[average_scores[0]/i, average_scores[1]/i],
            "times":(perf_counter() - runtime)/2
        })
    print("time total: ", perf_counter() - runtime)


# In[103]:


scores


# In[126]:


#you can see the val_acc and val loss are never able to get to a good place. maybe not enough capacity
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=.0001), loss='binary_crossentropy', metrics=['accuracy'])


# In[127]:


t = perf_counter()
history = model.fit(X_train, y_train, epochs=300, verbose=1, validation_split=0.2 )
print("train time", perf_counter()-t)


# In[123]:


t = perf_counter()
plot_loss_accuracy(history)
y_train

print(model.evaluate(X_test, y_test))
print("evaluation time", perf_counter()-t)

