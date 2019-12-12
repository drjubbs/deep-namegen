# -*- coding: utf-8 -*-
"""
Data preprocessing. Upcase everything, remove special characters "^" and "$" 
if they exist in the names. Encode the inputs and outputs for NN training 
(add details in this description).

TODO:
- Replace references to "cities"
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from plotly.offline import plot

WINDOW=6
PADDING="".join((WINDOW)*["^"])
MAX_LENGTH = 30
LETTERS="^ABCDEFGHIJKLMNOPQRSTUVWXYZ' -$"

def encode_in_out(x, y):
    x_matrix=np.zeros([len(x),len(LETTERS)])
    for i in range(len(x)):
        x_matrix[i][LETTERS.index(x[i])]=1
        
    x_vector=x_matrix.reshape(len(LETTERS)*len(x))    
        
    y_vector=np.zeros([len(LETTERS)])
    y_vector[LETTERS.index(y)]=1
    return(x_vector, y_vector)


def create_input_output(name_list):
    """
    Using context window, we'll pre-pad everything with starting characters 
    and a stopping character.
    
    Returns a human readable data frame, and one-hot encoded
    input/output pairs.
    """

    name_pad=[PADDING+t+"$" for t in name_list]

    x_human=[]
    X_list=[]
    y_human=[]
    y_list=[]
        
    for thisname in name_pad:
        for i in range(len(thisname)-WINDOW):
            x_human.append(thisname[i:i+WINDOW])
            y_human.append(thisname[i+WINDOW])
            
            x_mat, y_vec = encode_in_out(x_human[-1], y_human[-1])
            
            # Add on positional marker to first position
            x_mat=np.concatenate([np.array([i/MAX_LENGTH]), x_mat])
                        
            X_list.append(x_mat)
            y_list.append(y_vec)
            
    df=pd.DataFrame({
            'input' : x_human,
            'target' : y_human
            })
        
    return(df, np.array(X_list), np.array(y_list))
    
def run():
    with open("in/us_cities.txt", "r") as f:
        txt=f.read().split("\n")
    
    # Loop through all cities, make uppercase, skip
    # cities having a backslash
    cities=[]
    letters=set()
    for line in txt:
        t=line.upper().replace("^","").replace("$","")
        if not("/" in t):
            cities.append(t)
            letters=letters.union(set(t))
    
    # Make sure cities are unique
    cities=list(set(cities))

    # Histogrm of word length for reference
    len_results=[len(t) for t in cities]
 
    df=pd.DataFrame(len_results, columns=['length'])    
    fig=px.histogram(df , x='length', title="Database")
    fig.update_xaxes(range=[0, 30])
    fig.update_yaxes(title="")
    fig.update_layout(
        autosize=False,
        margin=dict(l=20, r=20, t=30, b=20),
        width=300,
        height=300)    
    plot(fig)
 
    if max(len_results)>MAX_LENGTH:
        raise(ValueError("Database contains excessively long name"))
    
    if max(len_results)<4:
        raise(ValueError("Database contains excessively short name"))
    
    
    # Make sure all the letters are in our set of 
    # encoded letters, excluding the special characters
    # for beginning pads and ending.
    if not(all([t in LETTERS[1:-1] for t in letters])):
        raise(ValueError("Letters present in names, missing in encoding."))
    
    # Split into training and test sets
    np.random.seed(20191125)
    idx=np.random.choice(list(range(len(cities))), size=len(cities),
                             replace=False)
    
    train=cities[0:len(idx)//4*3]
    test=cities[len(idx)//4*3:]
    
    # if we did this correctly, the intersection of the training and 
    # test sets should be zero
    if len(set(train).intersection(set(test)))>0:
        raise(ValueError("Problem with uniqueness of train/test"))
    
    
    # Look at histogram of lengths to determine size of context window
    plt.figure(1)    
    plt.hist([len(t) for t in cities], bins=[t+0.5 for t in range(15)])
    plt.show()
        
    if not os.path.exists('out'):
        os.makedirs('out')
     
    df_human_train, X_train, y_train=create_input_output(train)
    df_human_train.to_csv("./out/df_human_train.csv")
    
    df_human_test, X_test, y_test=create_input_output(test)
    df_human_test.to_csv("./out/df_human_test.csv")
    
    with open("./out/input.p","wb") as f:
        pickle.dump([X_train, y_train, X_test, y_test], f)




                
if __name__ == "__main__":
    run()