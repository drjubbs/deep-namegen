from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

model_dict = {}

"""
t = Sequential()
t.add(Dense(4, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model1']=t

t = Sequential()
t.add(Dense(8, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model2']=t

t = Sequential()
t.add(Dense(16, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model3']=t

t = Sequential()
t.add(Dense(128, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model4']=t

t = Sequential()
t.add(Dense(256, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model5']=t

t = Sequential()
t.add(Dense(512, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model6']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model7']=t

t = Sequential()
t.add(Dense(2048, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model8']=t

t = Sequential()
t.add(Dense(4096, activation='relu', input_dim=150))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['model9']=t


t = Sequential()
t.add(Dense(128, activation='relu', input_dim=150))
t.add(Dense(128, activation='relu'))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelA']=t


t = Sequential()
t.add(Dense(256, activation='relu', input_dim=150))
t.add(Dense(256, activation='relu'))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelB']=t

t = Sequential()
t.add(Dense(512, activation='relu', input_dim=150))
t.add(Dense(512, activation='relu'))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelC']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dense(1024, activation='relu'))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelD']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dropout(rate=0.1))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.1))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelE']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dropout(rate=0.2))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.2))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelF']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dropout(rate=0.3))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.3))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelG']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dropout(rate=0.4))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.4))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelH']=t

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dropout(rate=0.5))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelI']=t


t = Sequential()
t.add(Dense(2048, activation='relu', input_dim=150))
t.add(Dropout(rate=0.5))
t.add(Dense(2048, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelJ']=t

t = Sequential()
t.add(Dense(256, activation='relu', input_dim=150))
t.add(Dropout(rate=0.5))
t.add(Dense(256, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(256, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelK']=t


t = Sequential()
t.add(Dense(512, activation='relu', input_dim=150))
t.add(Dropout(rate=0.5))
t.add(Dense(512, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(512, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelL']=t
"""

t = Sequential()
t.add(Dense(1024, activation='relu', input_dim=150))
t.add(Dropout(rate=0.5))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(1024, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelM']=t

"""
t = Sequential()
t.add(Dense(2048, activation='relu', input_dim=150))
t.add(Dropout(rate=0.5))
t.add(Dense(2048, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(2048, activation='relu'))
t.add(Dropout(rate=0.5))
t.add(Dense(30, activation='softmax'))
t.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_dict['modelN']=t
"""