# -*- coding: utf-8 -*-
"""
Data preprocessing. Upcase everything, remove special characters "^" and "$"
if they exist in the names. Encode the inputs and outputs for NN training
(add details in this description).
"""

import pickle
import json
import base64
import gzip
import os
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import plot

# User parameters
ENCODE_FIRST_PROB = True
ENCODE_SECOND_PROB = True
LETTERS="^ABCDEFGHIJKLMNOPQRSTUVWXYZ'_-$"

class StatisticalProb:
    """Create probability table from targets.

    For the first two letters in a word sequence, we'll use the
    naturally occuring probabilities and joint probabilities if requested.
    This class builds and returns those vectors.
    """

    def __init__(self):
        """Constructor for class"""
        self._first_prob = None
        self._second_prob = None


    def calc_stats(self, words):
        """Calculate letter frequencies"""
        df_letters = pd.DataFrame({
                'first' : [t[0] for t in words],
                'second' : [t[1] for t in words]
                    })
        df_letters['one'] = 1

        # First letter probabilities
        cnts1 = df_letters[["first", "one"]].groupby("first").count()
        prob_first = cnts1 / cnts1.sum()
        for letter in LETTERS:
            if letter not in prob_first.index:
                prob_first.loc[letter, "one"]=0

        prob_first = prob_first.loc[list(LETTERS), :]
        self._first_prob = prob_first.copy()


        # Second letter probabilities conditoned on first
        cnts2 = df_letters.pivot_table(index='first', columns='second', aggfunc='count')
        cnts2.columns=[t[1] for t in cnts2.columns]

        for letter in LETTERS:
            if letter not in cnts2.index:
                cnts2.loc[letter, :]=0
            if letter not in cnts2.columns:
                cnts2.loc[:, letter]=0

        cnts2 = cnts2.loc[list(LETTERS), list(LETTERS)]
        cnts2 = cnts2.div(cnts2.sum(axis=1),axis=0)
        prob_second = cnts2.fillna(0)

        self._second_prob = prob_second.copy()


    def get_first_prob(self):
        """Return probability of first letter."""
        return self._first_prob.values.flatten()


    def get_second_prob(self, letter):
        """Return probability of second letter conditioned on first."""
        return self._second_prob.loc[letter,:].values


    def get_second_df(self):
        """Returns conditional probability as a matrix"""
        return self._second_prob


    def to_json(self):
        """Serialize to JSON"""
        json_dict = {}
        json_dict['_first_prob']=self._first_prob.to_json()
        json_dict['_second_prob']=self._second_prob.to_json()

        return json.dumps(json_dict)


    def from_json(self, json_txt):
        """Deserialize from JSON"""
        json_dict = json.loads(json_txt)
        self._first_prob = pd.read_json(json_dict['_first_prob'])
        self._second_prob = pd.read_json(json_dict['_second_prob'])

        # Fix order
        self._first_prob=self._first_prob.loc[list(LETTERS)]
        self._second_prob=self._second_prob.loc[list(LETTERS),list(LETTERS)]


class Preprocessor:
    """Preprocessor class, includes functions to do one-hot encoding of
    inputs and outputs.
    """

    def __init__(self):
        """Constructor, set some default attributes."""

        self._max_length = 0
        self._targets = None

        self.filename = None
        self.window = 0
        self.statistics = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def preprocess(self, input_filename, window):
        """Preprocesses input data. Strings are made upper case, spaces and
        special characters are stripped, spaces into underscores. Performs
        some data integrity checks. Data is split into training and test sets.
        Probability distributions of the first and second letter are calculated
        and cached.

        Human readable output of the input/output can be found in the following
        files:
        './out/df_human_train.csv'
        './out/df_human_test.csv'
        """

        self.filename = input_filename
        self.window = window

        with open(self.filename, "r") as input_file:
            txt = input_file.read().split("\n")

        self._max_length = 10 + max([len(t) for t in txt])

        # Loop through all cities, make uppercase, skip
        # cities having a backslash
        tgts=[]
        letters=set()
        for line in txt:
            txt=line.upper().replace("^","").replace("$","").replace(" ","_")
            if not "/" in txt:
                tgts.append(txt)
                letters=letters.union(set(txt))

        # Make sure targets are unique
        self._targets=list(set(tgts))

        # Nothing excessively long or short
        len_results=[len(t) for t in self._targets]
        if max(len_results)>self._max_length:
            raise ValueError("Database contains excessively long name")
        if max(len_results)<4:
            raise ValueError("Database contains excessively short name")

        # Make sure all the letters are in our set of
        # encoded letters, excluding the special characters
        # for beginning pads and ending.
        if not all([t in LETTERS[1:-1] for t in letters]):
            raise ValueError("Letters present in names missing in encoding.")

        # Split into training and test sets
        np.random.seed(20191125)
        idx=np.random.choice(list(range(len(self._targets))),
                                 size = len(self._targets),
                                 replace = False)

        train=self._targets[0:len(idx)//4*3]
        test=self._targets[len(idx)//4*3:]

        # if we did this correctly, the intersection of the training and
        # test sets should be zero
        if len(set(train).intersection(set(test)))>0:
            raise ValueError("Problem with uniqueness of train/test")

        # Create statistical table
        self.statistics = StatisticalProb()
        self.statistics.calc_stats(self._targets)

        # One hot encode and store in object
        df_human_train, self.x_train, self.y_train = \
            self._create_input_output(train)
        df_human_train.to_csv("./out/df_human_train.csv")

        df_human_test, self.x_test, self.y_test = \
            self._create_input_output(test)

        df_human_test.to_csv("./out/df_human_test.csv")


    def _encode_in_out(self, x_in, y_in, pos):
        """One-hot encoding of items.

        Convert text representation of x and y to one-hot encoding
        based on LETTERS global object.
        """
        x_matrix = np.zeros([len(x_in), len(LETTERS)])
        for i in enumerate(x_in):
            x_matrix[i[0]][LETTERS.index(i[1])] = 1

        # Prepend the positional encoding
        x_vector = x_matrix.reshape(len(LETTERS)*len(x_in))
        x_vector = np.concatenate([np.array(pos/self._max_length).reshape([1]),
                                     x_vector])

        y_vector=np.zeros([len(LETTERS)])
        y_vector[LETTERS.index(y_in)] = 1
        return (x_vector, y_vector)


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
        padding="".join((self.window)*["^"])
        name_pad = [padding+t+"$" for t in name_list]
        x_human = []
        x_list = []
        y_human=[]
        y_list = []

        for thisname in name_pad:
            for i in range(len(thisname) - self.window):
                x_human.append(thisname[i:i + self.window])
                y_human.append(thisname[i + self.window])

                x_mat, y_vec = self._encode_in_out(x_human[-1], y_human[-1], i)

                # Override first letter probabilities with table lookup
                if i == 0 and ENCODE_FIRST_PROB:
                    y_vec=self.statistics.get_first_prob()

                # Override second letter probabilities with table lookup,
                # this is conditional on the preceeding letter.
                if i == 1 and ENCODE_SECOND_PROB:
                    y_vec = self.statistics.get_second_prob(x_human[-1][-1])

                x_list.append(x_mat)
                y_list.append(y_vec)

        df_xy = pd.DataFrame({
                'input' : x_human,
                'target' : y_human
                })

        return(df_xy, np.array(x_list), np.array(y_list))


    def create_histogram(self):
        """Plot histogram of target lengths for reference."""

        len_results = [len(t) for t in self._targets]
        df_hist = pd.DataFrame(len_results, columns=['length'])
        fig = px.histogram(df_hist , x='length', title="Database")
        fig.update_xaxes(range = [0, 30])
        fig.update_yaxes(title = "")
        fig.update_layout(
            autosize = False,
            margin = dict(l = 20, r = 20, t = 30, b = 20),
            width = 600,
            height = 600)
        plot(fig)


    def get_starting_vector(self):
        """Returns a vector which represents the start of a new sequence.
        This is a vector of "^" (our padding character), which is the padding
        character, repeated `window` times.
        """

        # The "A" here is a dummy argument, we don't care about output.
        # Position is set to zero to denote start of string.
        x_init, _ = self._encode_in_out("".join(self.window*["^"]),"A", 0)

        # Skip the first entry, as this is the positional marker
        x_init = x_init[1:]
        x_init = x_init.reshape(1, len(LETTERS)*self.window)
        return x_init


    def get_max_length(self):
        """Return read-only attribute max_length"""
        return self._max_length


    def get_targets(self):
        """Return read-only attribute max_length"""
        return self._targets

    def to_json(self):
        """Serialize pre-processed object to JSON"""
        json_dict = {}
        json_dict['_max_length'] = self._max_length
        json_dict['filename'] = self.filename
        json_dict['window'] = self.window
        json_dict['statistics'] = self.statistics.to_json()

        json_dict['_targets'] = self._targets
        for field in ['x_train', 'x_test', 'y_train', 'y_test']:
            json_dict[field] = base64.b64encode(
                                gzip.compress(
                                pickle.dumps(
                                        self.__dict__[field]
                                ))).decode('ascii')

        return json.dumps(json_dict)


    def from_json(self, json_txt):
        """Deserialize from JSON string."""
        json_dict = json.loads(json_txt)
        self._max_length = json_dict['_max_length']
        self.filename = json_dict['filename']
        self.window = json_dict['window']
        stat_prob = StatisticalProb()
        stat_prob.from_json(json_dict['statistics'])
        self.statistics = stat_prob

        self._targets = json_dict['_targets']
        for field in ['x_train', 'x_test', 'y_train', 'y_test']:
            np_temp = pickle.loads(
                      gzip.decompress
                      (base64.b64decode(
                        json_dict[field]
                       )))
            self.__setattr__(field, np_temp)


if __name__ == "__main__":
    pre_proc=Preprocessor()

    # Filename, window size
    #pre_proc.preprocess("input/us_cities.txt", 7)
    pre_proc.preprocess("input/bible_characters.txt", 5)

    # Pickle and save
    if not os.path.exists('out'):
        os.makedirs('out')
    with open("./out/input.p","wb") as output_file:
        pickle.dump([pre_proc], output_file)

    txt_out=pre_proc.to_json()
    with open("./out/input.json","w") as output_file:
        output_file.writelines(txt_out)

    pre_proc.create_histogram()
    