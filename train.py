# -*- coding: utf-8 -*-
"""
Train the networks
"""
import pickle
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
import numpy as np
import time
import datetime as dt
import preprocessing as pp
import modelAZ as models

with open("input.p", "rb") as f:
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
                      epochs=100, batch_size=2000)
        
        train_err.append(log_loss(y_train[train_idx,:], 
                                   model.predict(X_train[train_idx,:])))
        out_of_bag.append(log_loss(y_train[val_idx,:], 
                                   model.predict(X_train[val_idx,:])))
    end_time=time.time()
    results[model_name] = (np.mean(train_err), 
                           np.mean(out_of_bag),
                           end_time-begin_time)

print(results)
fname = "models_{}".format(dt.datetime.now().strftime("%Y%m%d_%H%M"))
with open(fname, "wb") as f:
    pickle.dump(results, f)
print("Results saved as {}".format(fname))

# Refit last model and save
model.fit(X_train, y_train, epochs=250, batch_size=2000)
model.save("curr_model.h5")

"""
for i in range(40):
    actual=pp.LETTERS[np.argmax(y_train[i])]
    modeled=pp.LETTERS[np.argmax(model.predict(X_train[i].reshape(1,150)))]
    print(actual," ", modeled)
    
import matplotlib.pyplot as plt
model.history.history['accuracy']
plt.plot(model.history.history['accuracy'])
plt.show()
"""
