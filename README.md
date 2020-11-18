# Summary

This is a project to create a simple neural networks using Tensorflow/Keras which procedurally generates proper names for locations and people. Data sets available for experimentation (see `input` subdirectory):

- `us_cities.txt`: 29,880 US cities taken from the SQL database provided at https://github.com/kelvins/US-Cities-Database.git.
- `bible_characters.txt`: List of bible characters, adapted from https://www.wikidata.org/wiki/Wikidata:Lists/List_of_biblical_characters.

# Setup

Create a new virtual environment and install packages in `requirements.txt`.

Tensorflow works best with CUDA support. For `tensorflow 2.2`, this will be the slightly older 10.1 CUDA Development Kit from NVIDIA. Also add the full path  ( `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`) to your environment variables. Also the CuDNN package also must be installed and needs to match the same version as the developer toolkit.

You may also need to open the Display control panel (`Control Panel > Hardware and Sound > NVIDIA Control Panel > Manage 3D Settings` ) and under `Program Settings` force Python to use the GPU.

# Usage

1. Open `preprocessing.py` and modify the `main()` function to use one of the provided data sets (or provide your own).

2. Edit `models.py`, which returns `models_dict`, a dictionary of TensorFlow models, to include models for search. Follow the examples.

3. `param_search.py` loops over the models, and generates a pareto plot comparing overall training error to out of bag error (a decent measure of generalization). You can hover over the Plotly chart to manually select a model.

4. Open `final_fit.py`  to make novel names using the model. You may want to modify the parameter to `model.save()` to prevent the code from overwriting old models (and be sure the save the processor to a different filename as well  `./out/input.p`.

5. New names can be generated using the `generate.py` script. Ensure the model and preprocessor match the desired model:

   ```  model = tf.keras.models.load_model('curr_model.h5')
   with open("./out/input.p", "rb") as input_fn:
   	pp = pickle.load(input_fn)[0]
   ```

# Method Overview

Coming soon...

# TODO

- Use JSON for serialization instead of pickle.
- Switch to using command line arguments (vs. editing the files)
- Continue linting code and documentation
- Add an error rate prediction using the test set to the final model fitting
- Check/expand unit testing in `testing.py`

