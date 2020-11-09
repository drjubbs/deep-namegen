# Summary & Usage

This is a project to create a simple neural network using Tensorflow/Keras which procedurally generates proper names for locations and people. Data sets available for experimentation (see `input` subdirectory):

- `us_cities.txt`: 29,880 US cities taken from the SQL database provided at https://github.com/kelvins/US-Cities-Database.git.
- `bible_characters.txt`: List of bible characters, adapted from https://www.wikidata.org/wiki/Wikidata:Lists/List_of_biblical_characters.

1. Open `preprocessing.py` and change global variable `INPUT_FILE` to match one of the above files.
2. Edit `models.py`, which creates `models_dict`, a dictionary of TensorFlow models. 
3. `param_search.py` loops over the models, and generates a pareto plot comparing overall training error to out of bag error (a decent measure of generalization). You can hover over the Plotly chart to manually select a model.
4. Open `final_fit.py` and change global variable `MODEL_NAME` to the dictionary key of the selected model from the Pareto analysis. From this plot, it might be necessary to go back and tune the number of epochs used in the parameter search.
5. Run `generate.py` to make novel names using the model. Ensure the the preprocessor is set to the same input file:

```
...
usc=pp.Preprocessor("input/bible_characters.txt")
usc.preprocess()
...
```



# TODO

- Why are smaller batch sizes crashing? See `final_fit.py`... (see https://github.com/tensorflow/tensorflow/issues/44459)
- Store the preprocessor with the model so that `generate.py` doesn't come out of sync with the training.
- Modify `generate.py` to not repeat words already in the dictionary.

## Starting the Sequence

OPTIONAL ... The `y` vector for the first two letters of a name will be taken from the probability distribution.






