# -*- coding: utf-8 -*-
"""
Generate new names from fit models, excluding those already in the training/test
set. This is usually the last step in the workflow.
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


def parse_command_line():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fit and evaluate all models in `models.py`")
    parser.add_argument('label')
    parser.add_argument('model_name')
    opts = parser.parse_args(sys.argv[1:])

    # The fitting process generates a serialized preprocessor, read it in
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

    return opts, pre, model


def get_letter(model, x_pos):
    """Given a model, make a prediction and convert to specific letter using
    a random sampling method (i.e. model output is probability of the letter
    occurring).
    """

    prob = list(model.predict(x_pos)[0, :])

    # Normalize to 1 just to be extra safe even though softmax
    # should do this for us.
    prob = list(prob / sum(prob))

    a_prob = [0] + list(np.cumsum(prob[:-1]))
    b_prob = list(np.cumsum(prob[:-1])) + [1]

    prob_rand = np.random.rand()
    mask = [x1 <= prob_rand < x2 for x1, x2 in zip(a_prob, b_prob)]

    letter = LETTERS[mask.index(True)]

    return letter, mask


def generate_word(pre, model, model_name):
    """Generate a single word from the model. Note the `model_name` parameter
    is used to determine whether positional information is encoded in
    the vector.

    Inputs are the preprocessor object, the tensor flow model, and the
    model_name.
    """

    # Starting string without positional marker
    x_enc = pre.get_starting_vector()
    done = False
    length = 0
    word = []

    while not done:
        # LSTM networks don't need the positional information prepended to
        # the encoded sequence
        if model_name[0:4] == "LSTM":
            x_pos = x_enc.reshape(1, pre.window,
                                  int(x_enc.shape[1] / pre.window))
        elif model_name[0:4] == "DENS":
            # Augment with the positional indicator
            x_reshape = np.array([length / pre.get_max_length()]). \
                reshape(-1, 1)
            x_pos = np.concatenate([x_reshape, x_enc], axis=1)
        else:
            raise ValueError("Bad model type/prefix: %s" % model_name[0:4])

        letter, mask = get_letter(model, x_pos)

        if letter == "$":
            done = True
        else:
            # Drop the first letter ("roll-forward")
            stride = len(LETTERS)
            word.append(letter)
            x_enc = x_enc[:, stride:]

            # Append the new letter to the end
            new_letter_vec = np.zeros(stride).reshape(1, stride)
            new_letter_vec[0, mask.index(True)] = 1

            x_enc = np.concatenate([x_enc, new_letter_vec], axis=1)
            length = length + 1

    return "".join(word)


def main():
    """Main entry point"""
    opts, preprocessor, model = parse_command_line()

    num_words = 0
    # Create a histogram of new word lengths... this should be similar to
    # what was in the training set
    len_results = []

    while num_words < 100:

        new_word = generate_word(preprocessor, model, opts.model_name)

        # Skip if this word is in our training set...
        if new_word not in preprocessor.get_targets():
            num_words = num_words + 1
            print(new_word.replace("_", " ").title())
            len_results.append(len(new_word))

    # Plot histogram of new word lengths
    df_len = pd.DataFrame(len_results, columns=['length'])
    fig = px.histogram(df_len, x='length', title="Generated")
    fig.update_xaxes(range=[0, 30])
    fig.update_yaxes(title="")
    fig.update_layout(
        autosize=False,
        margin=dict(l=20, r=20, t=30, b=20),
        width=600,
        height=600)
    plot(fig)


if __name__ == "__main__":
    main()
