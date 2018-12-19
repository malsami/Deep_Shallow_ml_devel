import pickle
import torch
import logging

from sklearn.model_selection import train_test_split


def load_data():
    """
    Loads data from pickled files for fast access
    Returns
    -------
    tuple (List, List, List, List)
           data as tensors and as regular numpy arrays
       """

    # For Deep Learning
    data_tensor = pickle.load(open("x_tensor.p", "rb"))
    labels_tensor = pickle.load(open("y_tensor.p", "rb"))

    # Numpy array for sci-kit learn
    data = data_tensor.numpy()
    labels = labels_tensor.numpy()
    logging.info("Stored data is loaded ")

    return data_tensor, data, labels_tensor, labels


def train_val_test_split(x,y, train_split = .6, val_split = .2, test_split = .2):
    """
    Splits according to train,validation,test

    Parameters
    ----------
    x : numpy array [N x D]
        training set samples where n is number of samples and d is the dimension of each sample

    y : numpy array [N x 1]
        training set labels where each sample is either '1' or '0'.

    train_split : int optional
        how much of data set will be training data

    val_split : int optional
        how much of data will be validation data

    test_split : int optional
        how much of data will be test data

    Returns
    -------
    tuple (List, List, List, List, List, List)
        respective data for training, validation, and test
    """

    # Intermediate split between train and test data
    x_train, x_test, y_train, y_test = train_split(x, y, test_size=test_split, random_state=42, shuffle = True)

    # Split part of training set
    x_train, x_val, y_train, y_val = train_split(x_train, y_train, test_size=val_split, random_state=42, shuffle = True)

    return x_train, y_train, x_val, y_val, x_test, y_test

