# -*- coding: utf-8 -*-
"""Reproduce crash during fit"""
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


model = Sequential()
model.add(Dense(9, activation='relu', input_dim=125))
model.add(Dense(31, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
               metrics=['accuracy'])

X_train=np.random.rand(3225, 125)
y_train=np.random.rand(3225, 31)

# This works
model.fit(X_train[:,:], y_train[:,:], 
          epochs=100, 
          batch_size=X_train.shape[0], 
          verbose=1)

# This crashes
model.fit(X_train[:,:], y_train[:,:], 
          epochs=100, 
          batch_size=100, 
          verbose=1)