# -*- coding: utf-8 -*-
""" Hyper-parameter search using cross-fold validation. """

import os
import argparse
import sys
import time
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from preprocessing import Preprocessor
import models

PATIENCE = 10

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

def main():
    """Main entry point"""

    # Valiate command line args
    parser = argparse.ArgumentParser(
        description="Fit and evaluate all models in `models.py`")
    parser.add_argument('label')
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int)
    opts = parser.parse_args(sys.argv[1:])
    filename = os.path.join("output", opts.label+".json")

    if not os.path.exists(filename):
        parser.error("Could not find JSON preprocessed data: %s" % filename)

    # Create image output directory if it doesn't exist
    if not os.path.exists("images"):
        os.mkdir("images")

    # De-serialize preprocessor
    with open(filename, "r") as this_file:
        json_txt = this_file.read()
    pre_proc = Preprocessor()
    pre_proc.from_json(json_txt)

    # Create the models
    models_dict = models.generate_models(pre_proc)

    # Fitting with cross-validation
    kfolds = KFold(n_splits=models.K_FOLDS)
    results = {}

    # Loop over all models...
    for model_name, fold_models in models_dict.items():

        # Reshape X based on model type... standard neural networks
        # take a different shape than LSTM and have an additional input
        # for position in vector. This is handled automatically for LSTM
        # networks.
        if model_name[0:4]=="DENS":
            x_data = pre_proc.x_train
            y_data = pre_proc.y_train
        elif model_name[0:4]=="LSTM":
            x_data, y_data, _, _ = pre_proc.get_rnn_format()
        else:
            raise ValueError("Uknown model prefix: %s" % model_name[0:4])

        begin_time = time.time()
        train_err = []
        val_err = []
        history = []

        model_idx = 0
        for train_idx, val_idx in kfolds.split(x_data, y_data):

            # Grab the model for this fold
            model = fold_models[model_idx]

            train_dataset = tf.data.Dataset.from_tensor_slices((
                tf.cast(x_data[train_idx], tf.float32),
                tf.cast(y_data[train_idx], tf.float32),
            ))
            train_dataset = train_dataset.batch(opts.batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((
                tf.cast(x_data[val_idx], tf.float32),
                tf.cast(y_data[val_idx], tf.float32),
            ))
            val_dataset = val_dataset.batch(opts.batch_size)

            # No suffle, already done
            hist = model.fit(
                        x=train_dataset,
                        epochs=opts.epochs,
                        shuffle=False,              # Shuffle already done
                        verbose=0,
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        )

            # Story history and cycle to next fold
            history.append(hist)
            train_err.append(hist.history['loss'][-1])
            val_err.append(hist.history['val_loss'][-1])
            print("{0:10} {1:7.4f} {2:7.4f}".format(
                        model_name, train_err[-1], val_err[-1]))
            model_idx+=1

        # Done with all the folds
        end_time = time.time()
        fit_time = end_time-begin_time
        print("time: {0:7.2f}".format(fit_time))

        results[model_name] = (np.mean(train_err),
                            np.mean(val_err),
                            fit_time)

        fig = go.Figure()
        for i in range(models.K_FOLDS):
            fig.add_trace(go.Scatter(
                x = [t for t in range(len(history[i].history['loss']))],
                y = history[i].history['loss'],
                name = "Train Fold {}".format(i+1),
                mode = 'lines',
            ))
            fig.add_trace(go.Scatter(
                x = [t for t in range(len(history[i].history['val_loss']))],
                y = history[i].history['val_loss'],
                name = "Val Fold {}".format(i+1),
                mode = 'lines',
            ))
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title="Loss")
        outfile = os.path.join("images", "{0}_train_{1}.png".\
                                format(opts.label, model_name))
        fig.write_image(outfile, width=1920//2, height=1080//2)

    df_summary = pd.DataFrame(results).transpose()
    df_summary.columns = ['train', 'val', 'time']
    df_summary['labels'] = df_summary.index
    df_summary = df_summary[['labels', 'train', 'val', 'time']]

    outfile = os.path.join("output", "{0}_param_search.csv".format(opts.label))
    df_summary.to_csv(outfile)


if __name__ == "__main__":
    main()
