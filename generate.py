# -*- coding: utf-8 -*-
"""
Generate names from models
"""

import numpy as np
import tensorflow as tf
import preprocessing as pp
import pandas as pd


model = tf.keras.models.load_model('curr_model.h5')
vec_size = len(pp.LETTERS)*pp.WINDOW
stride = len(pp.LETTERS)

num_words=0    
while num_words<30:
    x = np.zeros(vec_size).reshape(1,vec_size) 
    length=0
    done = False
    word = []
    while done==False: 
        prob = list(model.predict(x)[0,:])
        
        A = ([0]+list(np.cumsum([0]+prob[1:])))[:-1]
        B = ([0]+list(np.cumsum([0]+prob[1:])))[1:]
        
        df=pd.DataFrame(zip(A,B), columns=["A", "B"], index=[t for t in pp.LETTERS])
                
        p = np.random.rand()
        mask=[p>=x1 and p<x2 for x1, x2 in zip(A,B)]
        
        letter=pp.LETTERS[mask.index(True)]
        if letter=="$":
            done=True
        else:
            word.append(letter)
            x=x[:,stride:]
            
            letter=np.zeros(stride).reshape(1,stride)
            letter[0,mask.index(True)]=1
            
            x=np.concatenate([x,letter], axis=1)
            length=length+1
    
    if len(word)>6:
        print("".join(word))
        num_words=num_words+1