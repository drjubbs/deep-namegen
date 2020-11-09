# -*- coding: utf-8 -*-
""" Final fitting of network after parameter search. """
import pickle
import time
from tensorflow import keras
import numpy as np
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import models
from preprocessing import Preprocessor, StatisticalProb

MODEL_NAME = "model0007"

# Load preprocessor object and model dictionary
with open("./out/input.p", "rb") as input_fn:
    pp = pickle.load(input_fn)[0]
model = models.generate_models(pp)[MODEL_NAME]

class LossHistory(keras.callbacks.Callback):
    """User defined extestion to keras callback to store training errors as
    a function of epoch."""
    def __init__(self):
        super().__init__()
        self.losses = None


    def on_train_begin(self, logs=None):
        self.losses = []


    def on_epoch_end(self, epoch, logs=None):
        if not logs is None:
            self.losses.append(logs.get('loss'))


history = LossHistory()

def logloss(y_true, y_pred, eps=1e-15):
    """We need our own log loss indicator because sci-kit does not
    support probabilities in y_true.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()

x_all=np.concatenate([pp.x_train, pp.x_test])
y_all=np.concatenate([pp.y_train, pp.y_test])

begin_time=time.time()
model.fit(x_all, y_all,
          epochs=500,
          batch_size=3000,
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
