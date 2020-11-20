# -*- coding: utf-8 -*-
"""
Generate new names from models, excluded those already in the training/test
set.
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from preprocessing import Preprocessor, LETTERS

def main():
    """Main entry point"""
    # Valiate command line args
    parser = argparse.ArgumentParser(
            description="Fit and evaluate all models in `models.py`")
    parser.add_argument('label')
    parser.add_argument('model_name')
    opts = parser.parse_args(sys.argv[1:])

    # Read preprocessor
    filename = os.path.join("output", opts.label+".json")
    if not os.path.exists(filename):
        parser.error("Could not find JSON preprocessed data: %s" % filename)
    with open(filename, "r") as this_file:
        json_txt = this_file.read()
    pre = Preprocessor()
    pre.from_json(json_txt)

    # Read model
    filename = os.path.join("output", "{0}_final_model.h5".format(opts.label))
    if not os.path.exists(filename):
        parser.error("Could not find model: %s" % filename)
    model = tf.keras.models.load_model(filename)

    stride = len(LETTERS)
    num_words = 0

    len_results = []
    while num_words<100:
        length = 0
        done = False
        word = []

        # Starting string without positional marker
        x_enc = pre.get_starting_vector()

        while not done:

            if opts.model_name[0:4]=="LSTM":
                x_pos = x_enc.reshape(1,pre.window,int(x_enc.shape[1]/pre.window))
            elif opts.model_name[0:4]=="DENS":
                # Augment with the positional indicator
                x_pos = np.concatenate([
                            np.array([length/pre.get_max_length()]).reshape(-1,1),
                            x_enc], axis=1)
            else:
                raise ValueError("Bad model type/prefix: %s" % opts.model_name[0:4])

            prob = list(model.predict(x_pos)[0,:])

            # Normalize to 1 just to be extra safe even though softmax
            # should do this for us.
            prob = list(prob/sum(prob))

            a_prob = [0]+list(np.cumsum(prob[:-1]))
            b_prob = list(np.cumsum(prob[:-1]))+[1]

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
        if test_output not in pre.get_targets():
            num_words=num_words+1
            print(test_output.replace("_", " ").title())
            len_results.append(len(word))

    # History of word lengths, this should match the input if everything
    # went OK.
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
