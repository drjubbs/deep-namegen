# -*- coding: utf-8 -*-
"""
Generates a dictionary of models which the parameter search will iterate
over. Models built using tensorflow with integrated Keras support.
"""
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import preprocessing as pp

OPTIMIZER = "Adam"
METRICS = ['categorical_crossentropy']

def generate_models(preprocess):
    """Return a dictionary of models to cycle through in the parameter 
    space search.
    """
    input_dim=len(pp.LETTERS)*preprocess.window+1
    output_dim=len(pp.LETTERS)

    model_dict = {}
        
    #---------------------------------------
    # DENS 000s Single layer
    #---------------------------------------
    params = [2**(t+2) for t in range(10)]
    base = 100
    for i, param in zip(range(len(params)), params):
        model = Sequential()
        model.add(Dense(param, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer=OPTIMIZER,
                    metrics=METRICS)
        model_dict['DENS{0:04d}'.format(i)]=model

    """
    #---------------------------------------
    # DENS 100s Two Layer
    #---------------------------------------
    params=[128, 256, 512, 1024, 2048, 3072, 4096]
    base=100
    for i, param in zip(range(len(params)), params):
        model = Sequential()
        model.add(Dense(param, activation='relu', input_dim=input_dim))
        model.add(Dense(param, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer=OPTIMIZER,
                    metrics=METRICS)
        model_dict['DENS{0:04d}'.format(i+base)]=model

    #---------------------------------------
    # DENS 200s Two Layers with Dropout
    #---------------------------------------
    params = [0.1, 0.2, 0.3, 0.4, 0.5]
    base=200
    for i, param in zip(range(len(params)), params):
        model = Sequential()
        model.add(Dense(2048, activation='relu', input_dim=input_dim))
        model.add(Dropout(rate=param))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(rate=param))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                optimizer=OPTIMIZER,
                metrics=METRICS)
        model_dict['DENS{0:04d}'.format(i+base)]=model
    
    #---------------------------------------
    # DENS 300s Three Layers with 0.5 dropout
    #---------------------------------------
    params = [2**(t+2) for t in range(10)]
    base=300
    for i, param in zip(range(len(params)), params):
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
        model_dict['DENS{0:04d}'.format(i+base)]=model

    #----------------------------------------
    # LSTM 000s Basic LSTM Model
    #-----------------------------------------
    params = [2**(t+2) for t in range(8)]
    base=000
    for i, param in zip(range(len(params)), params):
        model = Sequential()
        model.add(LSTM(param))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer=OPTIMIZER,
                    metrics=METRICS)
        model_dict['LSTM{0:04d}'.format(i+base)]=model    
    
    #----------------------------------------
    # LSTM 100s LSTM() + Dense
    #-----------------------------------------
    params = [2**(t+2) for t in range(10)]
    base=100
    for i, param in zip(range(len(params)), params):
        model = Sequential()
        model.add(LSTM(128))
        model.add(Dense(param))
        model.add(Dropout(rate=0.5))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer=OPTIMIZER,
                    metrics=METRICS)
        model_dict['LSTM{0:04d}'.format(i+base)]=model
    """

    #----------------------------------------
    # END
    #-----------------------------------------
    return model_dict
