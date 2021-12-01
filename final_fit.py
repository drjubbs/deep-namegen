# -*- coding: utf-8 -*-
""" Final fitting of network after parameter search. """
import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import models
from preprocessing import Preprocessor

EPOCHS = 5000
EARLY_STOPPING_FRACTION = 0.2
PATIENCE = 10

# Validate command line args
parser = argparse.ArgumentParser(
        description="Fit and evaluate all models in `models.py`")
parser.add_argument('label')
parser.add_argument('model_name')
parser.add_argument('batch_size', type=int)
opts = parser.parse_args(sys.argv[1:])

filename = os.path.join("output", opts.label+".json")
if not os.path.exists(filename):
    parser.error("Could not find JSON preprocessed data: %s" % filename)

# De-serialize preprocessor
with open(filename, "r") as this_file:
    json_txt = this_file.read()
pre = Preprocessor()
pre.from_json(json_txt)

# Reshape input based on model type
if opts.model_name[0:4] == "LSTM":
    xdata1, ydata1, xdata2, ydata2 = pre.get_rnn_format()
    x_all = np.concatenate([xdata1, xdata2])
    y_all = np.concatenate([ydata1, ydata2])
elif opts.model_name[0:4] == "DENS":
    x_all = np.concatenate([pre.x_train, pre.x_test])
    y_all = np.concatenate([pre.y_train, pre.y_test])
else:
    raise ValueError("Unknown network type: %s" % opts.model_name[0:4])

# Fetch the model, just take the first "fold" since we need
# only one model now
model_dict = models.generate_models(pre)
model = model_dict[opts.model_name][0]

# Setup early stopping...
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=PATIENCE,
        verbose=1,
    )
]

# Train the model
begin_time = time.time()
history = model.fit(x_all, y_all, epochs=5000, batch_size=opts.batch_size,
                    validation_split=0.2, callbacks=callbacks, verbose=1)
end_time = time.time()
print("Training time: {}".format(end_time-begin_time))

# Write to file
filename = os.path.join("output", "{0}_final_model.h5".format(opts.label))
model.save(filename)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(history.history['loss']))),
    y=history.history['loss'],
    name="Training",
    mode='lines',
))
fig.add_trace(go.Scatter(
    x=list(range(len(history.history['val_loss']))),
    y=history.history['val_loss'],
    name="Validation",
    mode='lines',
))
fig.update_xaxes(title="Epoch")
fig.update_yaxes(title="Loss")
plot(fig)
