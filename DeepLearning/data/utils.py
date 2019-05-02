import sys
import os
import sqlite3
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def read_sql_to_pandas(db_name):
    """
    reads database of given name
    always assumes tasksetsize of 3 (see sql_query below)

    Parameters
    ----------
    db_name: String

        Taskset sizes ranging from 1 to 3 (newer databases will support bigger sized tasksets

    Returns
    -------
    df: pandas dataframe
        Data Table of dataset in a nice easy to use/read dataframe
    """

    sql_query = ('select TaskSet.Set_ID, t1.*, t2.*, t3.*, TaskSet.Successful '
                         'from Task t1, Task t2, Task t3 '
                         'inner join TaskSet on t1.Task_ID = TaskSet.TASK1_ID '
                         'and '
                         't2.Task_ID = TaskSet.TASK2_ID '
                         'and ' 
                         't3.Task_ID = TaskSet.TASK3_ID '
                         'and TaskSet.TASK4_ID = -1 '
                        )

    print(db_name)
    conn = sqlite3.connect(db_name)

    df = pd.read_sql_query(sql_query, conn)

    conn.close()

    print("Loaded data into pandas database ")
    # Raw data for use
    return df


def clean_data (pandas_df):

    # make a copy for safety
    raw_df = pandas_df

    
    # COLUMNS TO DROP
    ID = ['Set_ID', 'Task_ID']
    CONSTANT_VALS = ['Deadline', 'CRITICALTIME', 'Quota', 'CAPS', 'PKG', 'CORES', 'COREOFFSET', 'OFFSET']
    
    

    print('Data is successfully cleaned and pickled')
    return raw_df.drop(columns=ID + CONSTANT_VALS)


def oversample_bad(df, factor=0):
    counter = 0
    value = [205891132094649, 847288609443, 2541865828329, 7625597484987, 22876792454961, 68630377364883]
    dropList = []
    bad = 0
    copies = []
    columns = list(df.columns)
    print(columns)
    newDF = df[~df.iloc[:,1].isin(value)]
    newDF = newDF[~newDF.iloc[:,5].isin(value)]
    newDF = newDF[~newDF.iloc[:,9].isin(value)]
    
    for index, data in newDF.iterrows():
        '''
        if data[1] in value or data [5] in value or data [9] in value:
            dropList.append(index)
            continue
        '''
        counter += 1
        if data[12]==0:
            single = []
            for point in data:
                single.append(point)
            for i in range(factor):
                copies.append(single)
            bad += 1

    #newDF.drop(dropList)
    #print('removed these mofos!', dropList)
            
    print('amount of duplicates:', len(copies))
    print('total:',counter,'bad:',bad)
    copiesDF = pd.DataFrame(copies, columns=columns)
    result = pd.concat([newDF,copiesDF], ignore_index=True)

    return result


def build_tensors(df, load_dataset=False):
    """
    Turn data into pytorch tensors for deep learning training. Also pickles to file for convenience later

    Parameters
    ----------
    df: pandas datarame
        data that is assumed to already be preprocessed
    load_dataset : bool optional
        Whether user wants to load in the pandas dataframe from before or pass his own in

    Returns
    -------
    tuple (List, List)
        Training Data and Labels as list of tensor objects
    """

    if load_dataset:
        df = pd.read_pickle('oversampledData.pkl')
        df = df.sample(frac=1).reset_index(drop=True)

    training_val = df
    y_tensor = torch.tensor(df['Successful'].values)
    training_val.drop(columns=['Successful'], axis=1, inplace=True)

    # Check pandas selecting all but one column
    normalized_data = normalize(training_val.values, axis=0)
    
    x_tensor = torch.tensor(normalized_data)

    pickle.dump(x_tensor, open('x_tensor.pkl', 'wb'))
    pickle.dump(y_tensor, open('y_tensor.pkl', 'wb'))

    print('Tensors created and saved')

    return x_tensor, y_tensor


def load_data():
    """
    Loads data from pickled files for fast access

    Returns
    -------
    tuple (List, List, List, List)
           data as tensors and as regular numpy arrays
       """

    # For Deep Learning
    print(os.path.dirname(os.path.realpath(__file__)))
    data_tensor = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/x_tensor.pkl', 'rb'))
    label_tensor = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/y_tensor.pkl', 'rb'))

    # Numpy array for sci-kit learn
    data = data_tensor.numpy()
    labels = label_tensor.numpy()
    print('Stored data is loaded')

    return data_tensor, data.astype(float), label_tensor, labels.astype(float)

def split_data():
    _, x, _, y = load_data()
    # Splits are by default .6 train, .2 val, .2 test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle = True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, shuffle = True)
    pickle.dump(x_train, open('split/x_train.pkl', 'wb'))
    pickle.dump(y_train, open('split/y_train.pkl', 'wb'))
    pickle.dump(x_test, open('split/x_test.pkl', 'wb'))
    pickle.dump(y_test, open('split/y_test.pkl', 'wb'))
    pickle.dump(x_val, open('split/x_val.pkl', 'wb'))
    pickle.dump(y_val, open('split/y_val.pkl', 'wb'))
    print('data was split')

def load_tensor_data():
    
    x_train = torch.Tensor(pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/split/x_train.pkl', 'rb')))
    y_train = torch.Tensor(pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/split/y_train.pkl', 'rb')))
    x_val = torch.Tensor(pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/split/x_val.pkl', 'rb')))
    y_val = torch.Tensor(pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/split/y_val.pkl', 'rb')))
    x_test = torch.Tensor(pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/split/x_test.pkl', 'rb')))
    y_test = torch.Tensor(pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/split/y_test.pkl', 'rb')))
    
    print('data was loaded')
    return x_train, y_train, x_val, y_val, x_test, y_test

def analyze():
    _, y_train, _, y_val, _, y_test = load_tensor_data()
    name = ['Training', 'Validation', 'Testing']
    data = [y_train, y_val, y_test]
    for i in range(3):
        total = 0
        bad = 0
        for success in data[i]:
            total += 1
            if success == 0:
                bad += 1
        print(name[i]+':\n', 'Total:',total, 'Bad:',bad,'bad/total:',(bad/total))

if __name__=='__main__':
    
    ### reads in raw data from sqlite db ###
    db_name = input('What is the name of the database file?')
    pandasDataFrame = read_sql_to_pandas(db_name)
    pandasDataFrame.to_pickle('raw_data.pkl')
    
    ### cleans up collumns ###
    cleanedData = clean_data(pandasDataFrame)
    cleanedData.to_pickle('cleanedData.pkl')
    print(cleanedData)
    ### oversamples bad class, before good:bad is about 4:1, with factor 3 its almost 1:1 ###
    oversampledData = oversample_bad(cleanedData, factor=3)
    oversampledData.to_pickle('oversampledData.pkl')
    print(oversampledData)
    ### builds data and label tensor ###
    x, y = build_tensors(None, load_dataset=True)

    ### splits data and label tensors into training:test:validation 3:1:1 ###
    split_data()

    
    ### tells absolute sizes of total, good and bad, and the proportion of bad/total ###
    analyze()
    