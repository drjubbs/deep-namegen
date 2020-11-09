# -*- coding: utf-8 -*-
"""
Generate names from models
"""

import numpy as np
import tensorflow as tf
import preprocessing as pp
import pandas as pd
import plotly.express as px

from plotly.offline import plot

model = tf.keras.models.load_model('curr_model.h5')
vec_size = len(pp.LETTERS)*pp.WINDOW
stride = len(pp.LETTERS)
len_results = [] 
usc=pp.Preprocessor(pp.INPUT_FILE)

#------------------------------------------
# Generate names
#------------------------------------------
num_words=0 
while num_words<1000:
     
    length = 0
    done = False
    word = []
    
    # Starting string without positional marker
    x = usc._encode_in_out("".join(pp.WINDOW*["^"]),"A", 0)[0][1:]
    x= x.reshape(1,vec_size)
       
    while done==False:

        # Augment with the positional indicator
        x_pos = np.concatenate([
                    np.array([length/pp.MAX_LENGTH]).reshape(-1,1),
                    x], axis=1)
        
        prob = list(model.predict(x_pos)[0,:])
        
        #if length<7:
        #   prob[pp.LETTERS.index("$")]=0
        
        prob=list(prob/sum(prob))
        
        A = ([0]+list(np.cumsum([0]+prob[1:])))[:-1]
        B = ([0]+list(np.cumsum([0]+prob[1:])))[1:]
        
        df=pd.DataFrame(zip(A,B), columns=["A", "B"], 
                        index=[t for t in pp.LETTERS])
                
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
        
    num_words=num_words+1
    print("".join(word).replace("_", " ").title())
        
    len_results.append(len(word))

df=pd.DataFrame(len_results, columns=['length'])    
fig=px.histogram(df , x='length', title="Generated")
fig.update_xaxes(range=[0, 30])
fig.update_yaxes(title="")
fig.update_layout(
    autosize=False,
    margin=dict(l=20, r=20, t=30, b=20),
    width=300,
    height=300)    
plot(fig)