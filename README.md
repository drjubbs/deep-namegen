# Summary

This is a project to create a simple neural networks using Tensorflow/Keras which procedurally generates proper names for locations and people. Data sets available for experimentation (see `input` subdirectory):

- `us_cities.txt`: 29,880 US cities taken from the SQL database provided at https://github.com/kelvins/US-Cities-Database.git.
- `bible_characters.txt`: List of bible characters, adapted from https://www.wikidata.org/wiki/Wikidata:Lists/List_of_biblical_characters.
- `testing.txt`: Simple test patterns used in unit testing.
- `counties.txt`: Counties in Pennsylvania USA, small example used for testing.

# Setup

Create a new virtual environment and install packages in `requirements.txt`.

Tensorflow works best with CUDA support. For `tensorflow 2.2`, this will be the slightly older 10.1 CUDA Development Kit from NVIDIA. Also add the full path  ( `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`) to your environment variables. Also the CuDNN package also must be installed and needs to match the same version as the developer toolkit.

You may also need to open the Display control panel (`Control Panel > Hardware and Sound > NVIDIA Control Panel > Manage 3D Settings` ) and under `Program Settings` force Python to use the GPU.

# Usage

1. Run `preprocessing.py` which takes two command line arguments. The first is a label which is the filename of the input data without the `.txt` extension. The second is an integer which specifies the window size to use.

2. Edit `models.py`, which returns `models_dict`, a dictionary of TensorFlow models, to include models for search. Follow the examples, comment or uncomment models as needed.

3. `param_search.py` loops over the models, and generates a pareto plot comparing overall training error to out of bag error (a decent measure of generalization). All network by network images and a summary chart are exported in `./images`. 

4. Open `final_fit.py`  to make one last model run top finalize.  `generate.py` then creates novel names using the model. Currently these functions are not setup to use command line arguments and must be modified manually.


# Method Overview

Coming soon...

# TODO

- Switch to using command line arguments (vs. editing the files) for the `final_fit.py` and `generate.py` scripts.
- Continue linting code and documentation, check/expand unit testing in `testing.py`
- Add an error rate prediction using the test set to the final model fitting. Translate the cross-entropy into something understandable to humans.

