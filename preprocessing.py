# -*- coding: utf-8 -*-
"""
Data preprocessing. Upcase everything, remove special characters "^" and "$" 
if they exist in the names. Encode the inputs and outputs for NN training 
(add details in this description).
"""
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from plotly.offline import plot

"""User parameters"""
WINDOW=6
ENCODE_FIRST_PROB = True
ENCODE_SECOND_PROB = True
MAX_LENGTH = 30
#INPUT_FILE=r"in/bible_characters.txt"
INPUT_FILE=r"in/us_cities.txt"

PADDING="".join((WINDOW)*["^"])
LETTERS="^ABCDEFGHIJKLMNOPQRSTUVWXYZ'_-$"

class StatisticalProb:
    """Create probability table from targets.
    
    For the first two letters in a word sequence, we'll use the
    naturally occuring probabilities and joint probabilities if requested.
    This class builds and returns those vectors.
    """
    
    def __init__(self, words): 
        df=pd.DataFrame({
                'first' : [t[0] for t in words],
                'second' : [t[1] for t in words]
                    })
        df['one']=1
        
        # First letter probabilities
        cnts1=df[["first","one"]].groupby("first").count()
        p1=cnts1/cnts1.sum()
        for l in LETTERS:
            if l not in p1.index:
                p1.loc[l,"one"]=0
        
        p1=p1.loc[[k for k in LETTERS],:]        
        self._first_prob=p1.copy()

        
        # Second letter probabilities conditoned on first
        cnts2=df.pivot_table(index='first', columns='second', aggfunc='count')
        cnts2.columns=[t[1] for t in cnts2.columns]
         
        for l in LETTERS:
            if l not in cnts2.index:
                cnts2.loc[l,:]=0
            if l not in cnts2.columns:
                cnts2.loc[:,l]=0
         
        cnts2=cnts2.loc[[k for k in LETTERS],[k for k in LETTERS]]
        cnts2=cnts2.div(cnts2.sum(axis=1),axis=0)
        p2=cnts2.fillna(0)
        
        self._second_prob=p2.copy()
        

    def get_first_prob(self):
        """Return probability of first letter."""
        return self._first_prob.values.flatten()                   

       
    def get_second_prob(self, letter):
        """Return probability of second letter conditioned on first."""
        return self._second_prob.loc[letter,:].values                
        
class Preprocessor:
    def _encode_in_out(self, x, y, pos):
        """One-hot encoding of items.
        
        Convert text representation of x and y to one-hot encoding
        based on LETTERS global object.
        """
        x_matrix=np.zeros([len(x),len(LETTERS)])
        for i in range(len(x)):
            x_matrix[i][LETTERS.index(x[i])]=1
   
        # Prepend the positional encoding         
        x_vector=x_matrix.reshape(len(LETTERS)*len(x))
        x_vector=np.concatenate([np.array(pos/MAX_LENGTH).reshape([1]), 
                                     x_vector])
             
        y_vector=np.zeros([len(LETTERS)])
        y_vector[LETTERS.index(y)]=1
        return(x_vector, y_vector)
    
    
    def _create_input_output(self, name_list):
        """Create encodings for all inputs.
        
        From a list of targets `name_list`, create X-y pairs using a context
        Window. Entries are pre-padded with starting characters and a stopping
        character. Global variables ENCODE_FIRST_PROB and ENCODE_SECOND_PROB
        realace the first and second `y` targest for a word with a probability
        instead of a hot encoding (i.e. when there is little to no context
        window available use a table lookp for next most likely)
          
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
                
                x_mat, y_vec = self._encode_in_out(x_human[-1], y_human[-1], i)
                
                # Override first letter probabilities with table lookup
                if i == 0 and ENCODE_FIRST_PROB:
                   y_vec=self.statistics.get_first_prob()
    
                # Override second letter probabilities with table lookup,
                # this is conditional on the preceeding letter.
                if i == 1 and ENCODE_SECOND_PROB:
                    y_vec=self.statistics.get_second_prob(x_human[-1][-1])
                            
                X_list.append(x_mat)
                y_list.append(y_vec)
                
        df=pd.DataFrame({
                'input' : x_human,
                'target' : y_human
                })
            
        return(df, np.array(X_list), np.array(y_list))
        
    def create_histogram(self):
        """Plot histogram of target lengths for reference.
        """
        len_results=[len(t) for t in self.targets]
        df=pd.DataFrame(len_results, columns=['length'])    
        fig=px.histogram(df , x='length', title="Database")
        fig.update_xaxes(range=[0, 30])
        fig.update_yaxes(title="")
        fig.update_layout(
            autosize=False,
            margin=dict(l=20, r=20, t=30, b=20),
            width=600,
            height=600)    
        plot(fig)
        
       
    def __init__(self, input_filename):
        with open(input_filename, "r") as f:
            txt=f.read().split("\n")
        
        # Loop through all cities, make uppercase, skip
        # cities having a backslash
        tgts=[]
        letters=set()
        for line in txt:
            t=line.upper().replace("^","").replace("$","").replace(" ","_")
            if not("/" in t):
                tgts.append(t)
                letters=letters.union(set(t))
        
        # Make sure targets are unique
        self.targets=list(set(tgts))
    
        # Nothing excessively long or short
        len_results=[len(t) for t in self.targets]    
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
        idx=np.random.choice(list(range(len(self.targets))), 
                                 size=len(self.targets),
                                 replace=False)
        
        train=self.targets[0:len(idx)//4*3]
        test=self.targets[len(idx)//4*3:]
        
        # if we did this correctly, the intersection of the training and 
        # test sets should be zero
        if len(set(train).intersection(set(test)))>0:
            raise(ValueError("Problem with uniqueness of train/test"))
                
        # Create statistical table
        self.statistics = StatisticalProb(self.targets)
            
        if not os.path.exists('out'):
            os.makedirs('out')
         
        df_human_train, X_train, y_train=self._create_input_output(train)
        df_human_train.to_csv("./out/df_human_train.csv")
        
        df_human_test, X_test, y_test=self._create_input_output(test)
        df_human_test.to_csv("./out/df_human_test.csv")
        
        with open("./out/input.p","wb") as f:
            pickle.dump([X_train, y_train, X_test, y_test], f)
                
if __name__ == "__main__":
    pp=Preprocessor(INPUT_FILE)
    pp.create_histogram()