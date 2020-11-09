# -*- coding: utf-8 -*-
"""
Generates a dictionary of models which the parameter search will iterate
over. Models built using tensorflow with integrated Keras support.
"""
import copy
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import preprocessing as pp


def generate_models(preprocess):
    """Give a window size, returns a diction of models to cycle through
    in the parameter space search.
    """

    input_dim=len(pp.LETTERS)*preprocess.window+1
    output_dim=len(pp.LETTERS)

    model_dict = {}
    base_model = Sequential()

    #---------------------------------------
    # 000s Single layer
    #---------------------------------------
    params = [2**(t+2) for t in range(10)]
    base = 100
    for i, param in zip(range(len(params)), params):
        model = copy.copy(base_model)
        model.add(Dense(param, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        model_dict['model{0:04d}'.format(i)]=model

    #---------------------------------------
    # 100s Two Layer
    #---------------------------------------
    params=[128, 256, 512, 1024, 2048, 3072, 4096]
    base=100
    for i, param in zip(range(len(params)), params):
        model = copy.copy(base_model)
        model.add(Dense(param, activation='relu', input_dim=input_dim))
        model.add(Dense(param, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        model_dict['model{0:04d}'.format(i+base)]=model

    #---------------------------------------
    # 200s Two Layers with Dropout
    #---------------------------------------
    params = [0.1, 0.2, 0.3, 0.4, 0.5]
    base=200
    for i, param in zip(range(len(params)), params):
        model = copy.copy(base_model)
        model.add(Dense(2048, activation='relu', input_dim=input_dim))
        model.add(Dropout(rate=param))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(rate=param))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        model_dict['model{0:04d}'.format(i+base)]=model

    #---------------------------------------
    # 300s Three Layers with 0.5 dropout
    #---------------------------------------
    params = [2**(t+2) for t in range(10)]
    base=300
    for i, param in zip(range(len(params)), params):
        model = copy.copy(base_model)
        model.add(Dense(param, activation='relu', input_dim=input_dim))
        model.add(Dropout(rate=0.5))
        model.add(Dense(param, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(param, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        model_dict['model{0:04d}'.format(i+base)]=model

    return model_dict
