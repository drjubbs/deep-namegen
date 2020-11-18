# -*- coding: utf-8 -*-
""" Hyper-parameter search using cross-fold validation. """

import os
import argparse
import sys
import time
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from preprocessing import Preprocessor
import models


def logloss(y_true, y_pred, eps=1e-15):
    """We need our own log loss indicator because sci-kit does not
    support probabilities in y_true.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()

def main():
    """Main entry point."""

    # Valiate command line args
    parser = argparse.ArgumentParser(
        description="Fit and evaluate all models in `models.py`")
    parser.add_argument('label')
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int)
    opts = parser.parse_args(sys.argv[1:])
    filename = os.path.join("output", opts.label+".json")

    if not os.path.exists(filename):
        parser.error("Could not fine JSON preprocessed data: %s" % filename)

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
    kfolds = KFold(n_splits=3)
    results = {}

    # Loop over all models...
    for model_name, model in models_dict.items():

        # Reshape X based on model type... standard neural networks
        # take a different shape than LSTM and have an additional input
        # for position in vector. This is handled automatically for LSTM
        # networks.
        if model_name[0:4]=="DENS":
            x_train = pre_proc.x_train
            y_train = pre_proc.y_train
        elif model_name[0:4]=="LSTM":
            x_train, y_train, _, _ = pre_proc.get_rnn_format()

        begin_time = time.time()
        out_of_bag = []
        train_err = []
        for train_idx, val_idx in kfolds.split(x_train, y_train):

            # Make sure batch size not larger than training data
            batch_size = min([opts.batch_size, len(train_idx)])

            history = model.fit(x_train[train_idx],
                      y_train[train_idx],
                      epochs=opts.epochs,
                      batch_size=batch_size,
                      verbose=0)

            llt = logloss(y_train[train_idx], model.predict(x_train[train_idx]))
            train_err.append(llt)
            llv = logloss(y_train[val_idx], model.predict(x_train[val_idx]))
            out_of_bag.append(llv)
            print("{0:10} {1:7.4f} {2:7.4f}".format(model_name, llt, llv))

        end_time = time.time()
        results[model_name] = (np.mean(train_err),
                            np.mean(out_of_bag),
                            end_time-begin_time)

        df_loss = {
            'epoch' : [t for t in range(len(history.history['loss']))],
            'loss' : history.history['loss'],
        }
        fig = px.scatter(data_frame=df_loss, x='epoch', y='loss')
        outfile = os.path.join("images", "{0}_train_{1}.png".\
                                format(opts.label, model_name))
        fig.write_image(outfile)


    df_summary = pd.DataFrame(results).transpose()    
    df_summary.columns = ['train', 'oob', 'time']
    df_summary['labels'] = df_summary.index
    df_summary = df_summary[['labels', 'train', 'oob', 'time']]

    fig = px.scatter(df_summary, x='train', y='oob', text='labels')
    outfile = os.path.join("images", "{0}_summary_{1}.png".\
                                format(opts.label, model_name))
    fig.write_image(outfile, width=1920, height=1080)


if __name__ == "__main__":
    main()

