from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import preprocessing as pp
import copy

INPUT_DIM=len(pp.LETTERS)*pp.WINDOW+1
OUTPUT_DIM=len(pp.LETTERS)

model_dict = {}
base_model = Sequential()

#---------------------------------------
# 000s Single layer
#---------------------------------------
params = [2**(t+2) for t in range(10)]
base = 100
for i, p in zip(range(len(params)), params):
    t = copy.copy(base_model)
    t.add(Dense(p, activation='relu', input_dim=INPUT_DIM))
    t.add(Dense(OUTPUT_DIM, activation='softmax'))
    t.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model_dict['model{0:04d}'.format(i)]=t


#---------------------------------------
# 100s Two Layer
#---------------------------------------
params=[128, 256, 512, 1024, 2048, 3072, 4096]
base=100
for i, p in zip(range(len(params)), params):
    t = copy.copy(base_model)
    t.add(Dense(p, activation='relu', input_dim=INPUT_DIM))
    t.add(Dense(p, activation='relu'))
    t.add(Dense(OUTPUT_DIM, activation='softmax'))
    t.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model_dict['model{0:04d}'.format(i+base)]=t

#---------------------------------------
# 200s Two Layers with Dropout
#---------------------------------------

params = [0.1, 0.2, 0.3, 0.4, 0.5]
base=200
for i, p in zip(range(len(params)), params):
    t = copy.copy(base_model)
    t.add(Dense(2048, activation='relu', input_dim=INPUT_DIM))
    t.add(Dropout(rate=p))
    t.add(Dense(2048, activation='relu'))
    t.add(Dropout(rate=p))
    t.add(Dense(OUTPUT_DIM, activation='softmax'))
    t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model_dict['model{0:04d}'.format(i+base)]=t

#---------------------------------------
# 300s Three Layers with 0.5 dropout
#---------------------------------------

params = [2**(t+2) for t in range(10)]
base=300
for i, p in zip(range(len(params)), params):
    t = copy.copy(base_model)
    t.add(Dense(p, activation='relu', input_dim=INPUT_DIM))
    t.add(Dropout(rate=0.5))
    t.add(Dense(p, activation='relu'))
    t.add(Dropout(rate=0.5))
    t.add(Dense(p, activation='relu'))
    t.add(Dropout(rate=0.5))
    t.add(Dense(OUTPUT_DIM, activation='softmax'))
    t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model_dict['model{0:04d}'.format(i+base)]=t
