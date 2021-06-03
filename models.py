# -*- coding: utf-8 -*-
"""
Generates a dictionary of models which the parameter search will iterate
over. Models built using tensorflow with integrated Keras support. Note
that each fold requires a new model -- otherwise Keras will remmeber the
weights from the previous fitting and not give a good indication of
generalization error.
"""
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import preprocessing as pp

# pylint: disable=no-name-in-module, too-many-statements

# Fitting parameters, adjust as needed
K_FOLDS = 3
OPTIMIZER = "Adam"
METRICS = ['categorical_crossentropy']


def generate_models(preprocess):
    """Return a dictionary of models to cycle through in the parameter
    space search.
    """
    input_dim = len(pp.LETTERS) * preprocess.window + 1
    output_dim = len(pp.LETTERS)

    model_dict = {}

    # ---------------------------------------
    # DENS 000s Single layer
    # ---------------------------------------
    params = [4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for i, param in zip(range(len(params)), params):
        models = []
        for _ in range(K_FOLDS):
            model = Sequential()
            model.add(Dense(param, activation='relu', input_dim=input_dim))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=METRICS)
            models.append(model)
        model_dict['DENS{0:04d}'.format(i)] = models

    # ---------------------------------------
    # DENS 100s Two Layer
    # ---------------------------------------
    params = [4, 16, 32, 64, 128, 256, 512]
    base = 100
    for i, param in zip(range(len(params)), params):
        models = []
        for _ in range(K_FOLDS):
            model = Sequential()
            model.add(Dense(param, activation='relu', input_dim=input_dim))
            model.add(Dense(param, activation='relu'))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=METRICS)
            models.append(model)
        model_dict['DENS{0:04d}'.format(i + base)] = models

    # ---------------------------------------
    # DENS 200s Two Layers with Dropout
    # ---------------------------------------
    params = [0.1, 0.2, 0.3, 0.4, 0.5]
    base = 200
    for i, param in zip(range(len(params)), params):
        models = []
        for _ in range(K_FOLDS):
            model = Sequential()
            model.add(Dense(2048, activation='relu', input_dim=input_dim))
            model.add(Dropout(rate=param))
            model.add(Dense(2048, activation='relu'))
            model.add(Dropout(rate=param))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=METRICS)
            models.append(model)
        model_dict['DENS{0:04d}'.format(i + base)] = models

    # ---------------------------------------
    # DENS 300s Three Layers with 0.5 dropout
    # ---------------------------------------
    params = [2 ** (t + 2) for t in range(10)]
    base = 300
    for i, param in zip(range(len(params)), params):
        models = []
        for _ in range(K_FOLDS):
            model = Sequential()
            model.add(Dense(param, activation='relu', input_dim=input_dim))
            model.add(Dropout(rate=0.5))
            model.add(Dense(param, activation='relu'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(param, activation='relu'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=METRICS)
            models.append(model)
        model_dict['DENS{0:04d}'.format(i + base)] = models

    # ----------------------------------------
    # LSTM 000s Basic LSTM Model
    # -----------------------------------------
    params = [2 ** (t + 2) for t in range(8)]
    base = 000
    for i, param in zip(range(len(params)), params):
        models = []
        for _ in range(K_FOLDS):
            model = Sequential()
            model.add(LSTM(param))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=METRICS)
            models.append(model)
        model_dict['LSTM{0:04d}'.format(i + base)] = models

    # ----------------------------------------
    # LSTM 100s LSTM() + Dense
    # -----------------------------------------
    params = [2 ** (t + 2) for t in range(8)]
    base = 100
    for i, param in zip(range(len(params)), params):
        models = []
        for _ in range(K_FOLDS):
            model = Sequential()
            model.add(LSTM(64))
            model.add(Dense(param))
            model.add(Dropout(rate=0.5))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=METRICS)
            models.append(model)
        model_dict['LSTM{0:04d}'.format(i + base)] = models
    # ----------------------------------------
    # END
    # -----------------------------------------

    return model_dict
