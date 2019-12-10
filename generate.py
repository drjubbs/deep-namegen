# -*- coding: utf-8 -*-
"""
Generate names from models
"""

import numpy as np
import tensorflow as tf
import preprocessing as pp
import pandas as pd


model = tf.keras.models.load_model('curr_model.h5')
    
for i in range(30):
    x = np.zeros(150).reshape(1,150) 
    length=0
    done = False
    while done==False: 
        prob = list(model.predict(x)[0,:])
        
        A = ([0]+list(np.cumsum([0]+prob[1:])))[:-1]
        B = ([0]+list(np.cumsum([0]+prob[1:])))[1:]
        
        df=pd.DataFrame(zip(A,B), columns=["A", "B"], index=[t for t in pp.LETTERS])
        
        p = np.random.rand()
        mask=[p>=x1 and p<x2 for x1, x2 in zip(A,B)]
        
        letter=pp.LETTERS[mask.index(True)]
        if length<=6 and letter=="$":
            pass 
        elif length>6 and letter=="$":
            print()
            done=True
        else:
            print(letter, end="")
            x=x[:,30:]
            
            letter=np.zeros(30).reshape(1,30)
            letter[0,mask.index(True)]=1
            
            x=np.concatenate([x,letter], axis=1)
            length=length+1