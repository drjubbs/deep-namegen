# -*- coding: utf-8 -*-
""" Hyper-parameter search using cross-fold validation. """

import pickle
import time
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from preprocessing import Preprocessor, StatisticalProb
import models

def logloss(y_true, y_pred, eps=1e-15):
    """We need our own log loss indicator because sci-kit does not
    support probabilities in y_true.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()

with open("./out/input.p", "rb") as input_fn:
    pp = pickle.load(input_fn)[0]

x_train = pp.x_train
y_train = pp.y_train
X_test = pp.x_test
y_test = pp.y_test

# Fitting with cross-validation
kf = KFold(n_splits=3)
results = {}

begin_all_time = time.time()
models_dict = models.generate_models(pp)

for model_name, model in models_dict.items():
    begin_time = time.time()
    out_of_bag = []
    train_err = []
    for train_idx, val_idx in kf.split(x_train, y_train):

        batch_size = min([2000, len(train_idx)])
        model.fit(x_train[train_idx, :], y_train[train_idx, :],
                  epochs=250, batch_size=batch_size, verbose=0)

        llt = logloss(y_train[train_idx, :],
                      model.predict(x_train[train_idx, :]))
        train_err.append(llt)
        llv = logloss(y_train[val_idx, :], model.predict(x_train[val_idx, :]))
        out_of_bag.append(llv)
        print(model_name, llt, llv)

    end_time = time.time()
    results[model_name] = (np.mean(train_err),
                           np.mean(out_of_bag),
                           end_time-begin_time)

labels = []
train = []
oob = []

for k, v in results.items():
    labels.append(k)
    train.append(v[0])
    oob.append(v[1])

df = pd.DataFrame({
                    'labels': labels,
                    'train': train,
                    'oob': oob,
                  })

fig = px.scatter(data_frame=df, x='train', y='oob', hover_name='labels')
plot(fig)
