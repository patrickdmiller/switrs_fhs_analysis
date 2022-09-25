#!/usr/bin/env python
# coding: utf-8

# In[109]:


import import_ipynb
import importlib
import framingham_data
importlib.reload(framingham_data)


# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from time import perf_counter
from sklearn.model_selection import StratifiedKFold, learning_curve
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[106]:


#function used for finding best score on hyperparameter tuning
def best_score(scores, SCORE_TYPE):
    best = {key: {"label":"", "score":0} for key in SCORE_TYPE}    
    for i in range(len(scores)):
        for s in SCORE_TYPE:
            if scores[i]['scores'][s] > best[s]["score"]:
                best[s]["label"] = scores[i]['label']
                best[s]["score"] = scores[i]['scores'][s]
    return best
        


# In[110]:


#load data with oversampling
X_train, y_train, X_test, y_test = framingham_data.load(oversample=True, verbose=True)
SCORE_TYPE = ["accuracy", "f1", "precision","recall"]


# In[87]:


'''
      _           _     _               _______                 
     | |         (_)   (_)             |__   __|                
   __| | ___  ___ _ ___ _  ___  _ __      | |_ __ ___  ___  ___ 
  / _` |/ _ \/ __| / __| |/ _ \| '_ \     | | '__/ _ \/ _ \/ __|
 | (_| |  __/ (__| \__ \ | (_) | | | |    | | | |  __/  __/\__ \
  \__,_|\___|\___|_|___/_|\___/|_| |_|    |_|_|  \___|\___||___/
                                                                
'''
#first create decision tree with varying depth
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
        scores[-1]["scores"][s] = sum(cross_val_score(clf, X_train, y_train,scoring=s, cv=10, n_jobs=6))/10
        scores[-1]["times"][s] = perf_counter() - t


# In[88]:


max_depth_scores = scores
print(best_score(max_depth_scores, SCORE_TYPE))
max_depth_scores


# In[91]:


#max features
scores = []
for i in range(1,20):
    clf = DecisionTreeClassifier(
        random_state=42,
        criterion="entropy",
        max_features=i,
        max_depth=25
    )
    scores.append({
        "label":f'max_features_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(clf, X_train, y_train,scoring=s, cv=10, n_jobs=6))/10
        scores[-1]["times"][s] = perf_counter() - t


# In[92]:


max_features_scores = scores
print(best_score(max_features_scores, SCORE_TYPE))
max_features_scores


# In[93]:


#print the results (used for graphing as well)
X_train, y_train, X_test, y_test = framingham_data.load(oversample=True, verbose=True, scale_type="ss")

estimator = DecisionTreeClassifier(
        random_state=42,
        max_depth=25,
        criterion="entropy",
        max_features=6
    )
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X_train, y_train, cv=10,return_times=True)
plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")


estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")
# plt.plot(y_pred, label="cross validation score")
plt.legend(loc="best")


# In[79]:


#test set on best
t = perf_counter()
best_depth = 23
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=best_depth,
    criterion="entropy",
    max_features=5
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
print(perf_counter() - t)


# In[80]:


'''
  _                     _   
 | |                   | |  
 | |__   ___   ___  ___| |_ 
 | '_ \ / _ \ / _ \/ __| __|
 | |_) | (_) | (_) \__ \ |_ 
 |_.__/ \___/ \___/|___/\__|
                            
                            
'''
from sklearn.ensemble import AdaBoostClassifier
X_train, y_train, X_test, y_test = framingham_data.load(oversample=True, verbose=True, scale_type="standard")


# In[105]:


# scores = []
# for i in range(10):
#     base = DecisionTreeClassifier(max_depth=1, criterion="entropy")
#     model = AdaBoostClassifier(base_estimator=base)
#     scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10, n_jobs=5)

#different starting estimators (weak learners, all 1 depth)
scores = []
SCORE_TYPE = ["accuracy", 'f1']

for i in range(1,110,10):
    base = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    model = AdaBoostClassifier(base_estimator=base,  n_estimators=(i * 50))
    scores.append({
        "label":f'n_estimators_{i*50}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(model, X_train, y_train, scoring=s, cv=10, n_jobs=6))/10
        scores[-1]["times"][s] = perf_counter() - t


# In[369]:


boost_n_estimators = scores
print(best_score(boost_n_estimators, SCORE_TYPE))
boost_n_estimators


# In[370]:


#the more learners the better our accuracy can we get the same performance with more levels and less learners

#different starting estimators (weak learners, all 1 depth)
scores = []
SCORE_TYPE = ["accuracy"]

for i in range(1,11):
    base = DecisionTreeClassifier(max_depth=i, criterion="entropy")
    model = AdaBoostClassifier(base_estimator=base,  n_estimators=(500))
    scores.append({
        "label":f'depth_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(model, X_train, y_train, scoring=s, cv=10, n_jobs=6))/10
        scores[-1]["times"][s] = perf_counter() - t


# In[371]:


boost_depth = scores
print(best_score(boost_depth, SCORE_TYPE))
boost_depth


# In[103]:


base = DecisionTreeClassifier(max_depth=10)
model = AdaBoostClassifier(base_estimator=base, n_estimators=5000)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=10,return_times=True, n_jobs=6)

plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")

model.fit(X_train, y_train)
t = perf_counter()
y_pred = model.predict(X_test)
print(perf_counter()-t)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")
# plt.plot(y_pred, label="cross validation score")
plt.legend(loc="best")


# In[373]:



t = perf_counter()
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test accuracy")
# plt.plot(y_pred, label="cross validation score")
plt.legend(loc="best")
print(perf_counter()-t)


# In[96]:


'''
  _____   ___ __ ___  
 / __\ \ / / '_ ` _ \ 
 \__ \\ V /| | | | | |
 |___/ \_/ |_| |_| |_|
                      
                      
'''
from sklearn.svm import SVC
X_train, y_train, X_test, y_test = framingham_data.load(oversample=True, verbose=True, scale=True, scale_type="standard")


# In[375]:


scores = []
SCORE_TYPE = ["accuracy"]
KERNELS = ['linear','poly', 'rbf','sigmoid']
for k in KERNELS:
    svm = SVC(kernel=k)

    scores.append({
        "label":f'kernel_{k}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(svm, X_train, y_train, scoring=s, cv=10, n_jobs=6))/10
        scores[-1]["times"][s] = perf_counter() - t


# In[376]:


svm_scores= scores
print(best_score(svm_scores, SCORE_TYPE))
svm_scores


# In[377]:


scores = []
SCORE_TYPE = ["accuracy"]
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


# In[378]:


svm_scores= scores
print(best_score(svm_scores, SCORE_TYPE))
svm_scores


# In[104]:


model = SVC(kernel="rbf", gamma = 1)
model.fit(X_train, y_train)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=10,return_times=True, n_jobs=6)

plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")


model.fit(X_train, y_train)
t = perf_counter()
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")
plt.legend(loc="best")
print(perf_counter()-t)


# In[150]:


svm = SVC(kernel="poly", degree=30)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))


# In[99]:


'''
  _                
 | |               
 | | ___ __  _ __  
 | |/ / '_ \| '_ \ 
 |   <| | | | | | |
 |_|\_\_| |_|_| |_|
                   
                   
'''

from sklearn.neighbors import KNeighborsClassifier
X_train, y_train, X_test, y_test = framingham_data.load(oversample=True, verbose=True, scale=True, scale_type="standard")


# In[398]:


scores = []
SCORE_TYPE = ["accuracy"]
for i in range(2, 20, 1):
    knn = KNeighborsClassifier(n_neighbors=i)

    scores.append({
        "label":f'knn_k_{i}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(knn, X_train, y_train, scoring=s, cv=10, n_jobs=6))/10
        scores[-1]["times"][s] = perf_counter() - t





# In[399]:


knn_k_scores= scores
print(best_score(knn_k_scores, SCORE_TYPE))
knn_k_scores


# In[196]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))


# In[395]:


scores = []
SCORE_TYPE = ["accuracy"]
algos= ['auto','ball_tree','kd_tree','brute']
weights = ["uniform","distance"]
ps = [1,2]
for l in range(5,150,10):
# #     for a in weights:
# for a in ps:
    print(l)
    knn = KNeighborsClassifier(n_neighbors=1, leaf_size=l)

    scores.append({
        "label":f'knn_l_{l}',
        "scores":{},
        "times":{}
    })
    for s in SCORE_TYPE:
        t = perf_counter()
        scores[-1]["scores"][s] = sum(cross_val_score(knn, X_train, y_train, scoring=s, cv=10))/10
        scores[-1]["times"][s] = perf_counter() - t
knn_k_scores = scores
knn_k_scores= scores
print(best_score(knn_k_scores, SCORE_TYPE))
knn_k_scores


# In[100]:


model = KNeighborsClassifier(n_neighbors=2)
# model.fit(X_train, y_train)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=10,return_times=True, n_jobs=6)

plt.plot(train_sizes,np.mean(train_scores,axis=1), label="training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1), label="cross validation score")
plt.xlabel("Num Samples")
plt.ylabel("Score")


model.fit(X_train, y_train)
t = perf_counter()
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("f1:",metrics.f1_score(y_test, y_pred))
plt.hlines(y=metrics.accuracy_score(y_test, y_pred), color='green', xmin=0, xmax = len(X_train), label="test accuracy")
plt.hlines(y=metrics.f1_score(y_test, y_pred), color='red', xmin=0, xmax = len(X_train), label="test f1")

plt.legend(loc="best")
print(perf_counter()-t)


# In[112]:


'''
                             _            _                      _    
                            | |          | |                    | |   
  _ __   ___ _   _ _ __ __ _| |_ __   ___| |___      _____  _ __| | __
 | '_ \ / _ \ | | | '__/ _` | | '_ \ / _ \ __\ \ /\ / / _ \| '__| |/ /
 | | | |  __/ |_| | | | (_| | | | | |  __/ |_ \ V  V / (_) | |  |   < 
 |_| |_|\___|\__,_|_|  \__,_|_|_| |_|\___|\__| \_/\_/ \___/|_|  |_|\_\
                                                                      
                                                                      
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X_train, y_train, X_test, y_test = framingham_data.load(oversample=True, verbose=True, scale=True, scale_type="ss")
from tensorflow.keras.optimizers import Adam
def train_validate_fold(model, _X_train, _y_train, _X_val, _y_val, epochs=200):
    model.get_weights()
    print("training for ", epochs, "epochs")
    model.fit(_X_train, _y_train, epochs=epochs, verbose = 0)
    scores = model.evaluate(_X_val, _y_val, verbose=0)
    return scores


# In[263]:


#you can see the val_acc and val loss are never able to get to a good place. maybe not enough capacity
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[259]:


history = model.fit(X_train, y_train, epochs=200, verbose=1, validation_split=0.2 )


# In[8]:



def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['accuracy'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))


# In[311]:


scores = []
nodes_to_try = [8,16,32,64]
for n in nodes_to_try:
    print( " -- ", n ,"--")
    model = Sequential()
    model.add(Dense(n, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
            "times":(perf_counter() - runtime)/5
        })


# In[312]:


print(scores)


# In[320]:


layer_test_scores = [{'label': 'nn_n_8', 'scores': [0.5139356553554535, 0.7608663320541382], 'times': 31.644711472501513}, {'label': 'nn_n_16', 'scores': [0.40235650539398193, 0.8605446815490723], 'times': 31.634031138999852}, {'label': 'nn_n_32', 'scores': [0.1801611315459013, 0.9620467305183411], 'times': 31.14031546320184}, {'label': 'nn_n_64', 'scores': [0.1330405578482896, 0.9776800036430359], 'times': 31.143438562803205}]


# In[319]:


# so from this, the best is 64 units which passes validation tests. 
#now lets alter the learning rate
scores = []
learning_rates = [.1, .01, .001, .0001]
for lr in learning_rates:
    print(" lr: ", lr)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
            "times":(perf_counter() - runtime)/5
        })


# In[321]:


print(scores)


# In[39]:


#final nn 64,64 with lr of .01 (which is best)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[40]:


t = perf_counter()
history = model.fit(X_train, y_train, epochs=300, verbose=0,  validation_split=0.2 )
print("train time", perf_counter()-t)


# In[41]:


t = perf_counter()
plot_loss_accuracy(history)
y_train

print(model.evaluate(X_test, y_test))
print("evaluation time", perf_counter()-t)

