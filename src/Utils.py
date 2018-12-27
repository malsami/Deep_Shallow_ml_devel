import pickle
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'C:\\Users\\Varun\\Documents\\Misc\\Research\\MalSami\\Deep_Shallow_ml_devel\\'


def load_data():
    """
    Loads data from pickled files for fast access

    Returns
    -------
    tuple (List, List, List, List)
           data as tensors and as regular numpy arrays
       """

    # For Deep Learning
    data_tensor = pickle.load(open(path + "data\\processed\\" + "x_tensor.p", "rb"))
    labels_tensor = pickle.load(open(path + "data\\processed\\" + "y_tensor.p", "rb"))

    # Numpy array for sci-kit learn
    data = data_tensor.numpy()
    labels = labels_tensor.numpy()
    logging.info("Stored data is loaded ")

    return data_tensor, data, labels_tensor, labels


def train_val_test_split(x, y, val_split=.2, test_split=.2, to_tensor = False):
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

    to_tensor : boolean optional
        Whether we want the arrays to be tensors or numpy arrays. Defaulted to numpy arrays

    Returns
    -------
    tuple (List, List, List, List, List, List)
        respective data for training, validation, and test
    """

    # Intermediate split between train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=42, shuffle = True)

    # Split part of training set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42, shuffle = True)

    if to_tensor :
        x_train = torch.Tensor(x_train)
        x_val = torch.Tensor(x_val)
        x_test = torch.Tensor(x_test)
        #######
        y_train = torch.Tensor(y_train)
        y_val = torch.Tensor(y_val)
        y_test = torch.Tensor(y_test)

    else:
        pass

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_tensors(clean_df, load_dataset=False):
    """
    Turn data into pytorch tensors for deep learning training. Also pickles to file for convenience later

    Parameters
    ----------
    clean_df: pandas datarame
        data that is assumed to already be preprocessed
    load_dataset : bool optional
        Whether user wants to load in the pandas dataframe from before or pass his own in

    Returns
    -------
    tuple (List, List)
        Training Data and Labels as list of tensor objects
    """

    if load_dataset:
        # clean_df = pickle.load(open("clean_raw_data.p","rb"))
        clean_df = pd.read_pickle(path + "data\\interim\\" + "clean_raw.pkl")

    training_val = clean_df
    y_tensor = torch.tensor(clean_df['Successful'].values)
    training_val.drop('Successful', axis=1)

    # Check pandas selecting all but one column
    x_tensor = torch.tensor(training_val.values)

    pickle.dump(x_tensor, open(path + "data\\processed\\" + "x_tensor.p", "wb"))
    pickle.dump(y_tensor, open(path + "data\\processed\\" + "y_tensor.p", "wb"))

    logging.info("Tensors created and saved")

    return x_tensor, y_tensor
