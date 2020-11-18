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

    model = tf.keras.models.load_model('./output/curr_model.h5')
    FILENAME = "./output/bible_characters.json"

    # De-serialize preprocessor
    with open(FILENAME, "r") as this_file:
        json_txt = this_file.read()
    pre = Preprocessor()
    pre.from_json(json_txt)

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
        x_enc = pre.get_starting_vector()

        while not done:

            # Augment with the positional indicator
            x_pos = np.concatenate([
                        np.array([length/pre.get_max_length()]).reshape(-1,1),
                        x_enc], axis=1)

            prob = list(model.predict(x_pos)[0,:])
            prob = list(prob/sum(prob))

            # Set up the high and low limits for each letter
            a_prob = ([0]+list(np.cumsum([0]+prob[1:])))[:-1]
            b_prob = ([0]+list(np.cumsum([0]+prob[1:])))[1:]

            prob_rand = np.random.rand()
            mask=[prob_rand>=x1 and prob_rand<x2 for x1, x2 in \
                        zip(a_prob, b_prob)]

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
        if not test_output in pre.get_targets():
            num_words=num_words+1
            print(test_output.replace("_", " ").title())
            len_results.append(len(word))

    df_len=pd.DataFrame(len_results, columns=['length'])
    fig=px.histogram(df_len , x='length', title="Generated")
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
