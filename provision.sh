#!/bin/bash

# Boring updates
sudo apt-get update -qq
sudo apt-get install python3.5
sudo apt-get -y install python3-pip
# sudo pip3 install --up
# sudo pip install 
# Regular imports
pip3 install -U numpy
pip3 install -U pickle

# Sk-learn important libraries 
pip3 install -U scikit-learn

# Deep Learning
#pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip3 install torch torchvision

# Tensorflow + Keras
pip3 install --upgrade tensorflow
pip3 install keras
# Visualization Purposes (May not be necessary here)
pip3 install pandas
pip3 install seaborn


