# Summary

This is a project to create a simple neural network using Tensorflow/Keras which procedurally generates proper names for locations and people. Data sets available for experimentation:

- `us_cities.txt`: 29,880 US cities taken from the SQL database provided at https://github.com/kelvins/US-Cities-Database.git.
- `bible_characters.txt`: List of bible characters, adapted from https://www.wikidata.org/wiki/Wikidata:Lists/List_of_biblical_characters.

# Setup

Create a new virtual environment and install packages in `requirements.txt`. (tensorflow, pandas, plotly, sklearn)

Tensorflow will require CUDA support. For `tensorflow 2.2`, this will be the slightly older 10.1 CUDA Development Kit from NVIDIA. Also add the full path to your environment variables `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`. Also the CuDNN package also must be installed and needs to match the same version as the developer toolkit.

You may also need to open the Display control panel and under "advanced" force "python.exe" to use the GPU.

## Starting the Sequence

OPTIONAL ... The `y` vector for the first two letters of a name will be taken from the probability distribution.






