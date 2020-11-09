# -*- coding: utf-8 -*-
"""
Generate new names from models, excluded those already in the training/test
set.
"""
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from preprocessing import Preprocessor, StatisticalProb, LETTERS

def main():
    """Main entry point"""

    model = tf.keras.models.load_model('curr_model.h5')
    with open("./out/input.p", "rb") as input_fn:
        pp = pickle.load(input_fn)[0]

    stride = len(LETTERS)
    len_results = [] 

    #------------------------------------------
    # Generate names
    #------------------------------------------
    num_words = 0
    while num_words<1000:

        length = 0
        done = False
        word = []

        # Starting string without positional marker
        x_enc = pp.get_starting_vector()

        while not done:

            # Augment with the positional indicator
            x_pos = np.concatenate([
                        np.array([length/pp._max_length]).reshape(-1,1),
                        x_enc], axis=1)

            prob = list(model.predict(x_pos)[0,:])
            prob = list(prob/sum(prob))

            A = ([0]+list(np.cumsum([0]+prob[1:])))[:-1]
            B = ([0]+list(np.cumsum([0]+prob[1:])))[1:]

            df=pd.DataFrame(zip(A,B),
                            columns=["A", "B"],
                            index=list(LETTERS))

            p = np.random.rand()
            mask=[p>=x1 and p<x2 for x1, x2 in zip(A,B)]

            letter=LETTERS[mask.index(True)]
            if letter=="$":
                done=True
            else:
                word.append(letter)
                x_enc=x_enc[:,stride:]

                letter=np.zeros(stride).reshape(1,stride)
                letter[0,mask.index(True)]=1

                x_enc=np.concatenate([x_enc,letter], axis=1)
                length=length+1

        # Skip if this word is in our training set...
        test_output="".join(word)
        if not test_output in pp._targets:
            num_words=num_words+1
            print(test_output.replace("_", " ").title())
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


if __name__ == "__main__":
    main()