# Deep_Shallow_ml_devel
Machine Learning Environment setup and Development

set up virtual environment with:
virtualenv --python=/usr/bin/python3.5 /path/to/venv

Once created, activate with
source ./venv/bin/activate 

Shallow Learning Models were adapted from Robert (LordNorbi).

Deep Learning Models were adapted from Robert Hamsch



## Machine Learning

The Machine Learning models can be found in /models/*

## Running

The scripts can be run via the python files or ipynb notebooks. The notebooks load the appropriate functions from the
*.py files. Jupyter notebooks are convenient because of the quick results and graphical interface. In terms of functionality
they are both the same. Notebooks are in the *notebooks* folder but are not fully configured yet. 

To retrieve the data and build files, put the *pandas* database in the *data/external* folder and then run the script 
*python DataRetrievalAndClean.py [taskset_size]* where *taskset_size* is a number from 1-3. 

The file *Trainer.py* is for machine learning training. See the file for more information. 
