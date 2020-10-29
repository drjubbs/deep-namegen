# -*- coding: utf-8 -*-
""" Final fitting of network after parameter search. """
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import models
import plotly.express as px
from plotly.offline import plot
import pandas as pd

MODEL_NAME = "model0009"
model = models.model_dict[MODEL_NAME]

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
history = LossHistory()

def logloss(y_true, y_pred, eps=1e-15):
    """We need our own log loss indicator because sci-kit does not
    support probabilities in y_true.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()

with open("./out/input.p", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)


begin_time=time.time()

# TODO: Small batch sizes crash ?!?!?! (less than 3k for bible names)
model.fit(X_train[:,:], y_train[:,:], 
          epochs=500, 
          batch_size=X_train.shape[0], 
          verbose=1, 
          callbacks=[history])

model.save("curr_model.h5")
end_time=time.time()

print("Training time: {}".format(end_time-begin_time))

# Loss vs. epoch
df=pd.DataFrame({'x' : range(len(history.losses)), 
                 'y' : history.losses})
fig=px.scatter(df, x='x', y='y')
fig.update_traces(marker=dict(size=12, line=dict(width=2,
                                        color='DarkSlateGrey')))
plot(fig)