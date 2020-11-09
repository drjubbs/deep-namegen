# -*- coding: utf-8 -*-
"""
Train the networks. At the end prints out a dictionary
which can be pasted in pareto.py to visualization train/validations
errors.
"""
# Debug - uncomment to turn off GPU support
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import numpy as np
import time
import datetime as dt
import modelAZ as models

# This controls which model is re-fit and then saved 
# at the end of the search. Set to 'none' if refitting
# is not desired.
SAVE_MODEL="model0009"


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

begin_all_time=time.time()

for model_name, model in models.model_dict.items():
    begin_time=time.time()
    out_of_bag=[]
    train_err = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        
        xt=X_train[train_idx,:]
        yt=y_train[train_idx,:]
                
        model.fit(X_train[train_idx,:], y_train[train_idx,:], 
                      epochs=100, batch_size=2000, verbose=0)
        
        llt=logloss(y_train[train_idx,:], model.predict(X_train[train_idx,:]))
        train_err.append(llt)
        llv=logloss(y_train[val_idx,:], model.predict(X_train[val_idx,:]))
        out_of_bag.append(llv)
        
        print(model_name, llt, llv)
        
    end_time=time.time()
    results[model_name] = (np.mean(train_err), 
                           np.mean(out_of_bag),
                           end_time-begin_time)

print(results)
end_all_time=time.time()
print("Total time:", end_all_time-begin_all_time)

if SAVE_MODEL is not None:
    # Refit last model and save
    model=models.model_dict[SAVE_MODEL]
    print(SAVE_MODEL+ " fit and saved.")
    model.fit(X_train, y_train, epochs=250, batch_size=2000, verbose=0)
    model.save("curr_model.h5")