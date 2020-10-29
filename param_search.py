# -*- coding: utf-8 -*-
""" Hyper-parameter search using cross-fold validation. """
import pickle
from sklearn.model_selection import KFold
import numpy as np
import time
import models
import pandas as pd
import plotly.express as px
from plotly.offline import plot

def logloss(y_true, y_pred, eps=1e-15):
    """We need our own log loss indicator because sci-kit does not
    support probabilities in y_true.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()

with open("./out/input.p", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# Fitting with cross-validation
kf = KFold(n_splits=3)
results = {}

for model_name, model in models.model_dict.items():
    begin_time=time.time()
    out_of_bag=[]
    train_err = []
    for train_idx, val_idx in kf.split(X_train, y_train):
                
        model.fit(X_train[train_idx,:], y_train[train_idx,:], 
                      epochs=100, batch_size=len(train_idx), verbose=0)
        
        llt=logloss(y_train[train_idx,:], model.predict(X_train[train_idx,:]))
        train_err.append(llt)
        llv=logloss(y_train[val_idx,:], model.predict(X_train[val_idx,:]))
        out_of_bag.append(llv)
        
        print(model_name, llt, llv)
        
    end_time=time.time()
    results[model_name] = (np.mean(train_err), 
                           np.mean(out_of_bag),
                           end_time-begin_time)

labels=[]
train=[]
oob=[]

for k,v in results.items():
    labels.append(k)
    train.append(v[0])
    oob.append(v[1])
    
df=pd.DataFrame({
    'labels' : labels,
    'train' : train,
    'oob' : oob,
    })
    
fig=px.scatter(data_frame=df, x='train', y='oob', hover_name='labels')
plot(fig)