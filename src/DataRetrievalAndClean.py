import sqlite3
import pandas as pd
import torch
from torch.utils.serialization import load_lua
import pickle
import logging
import sys

from src.Utils import build_tensors, load_data

path = 'C:\\Users\\Varun\\Documents\\Misc\\Research\\MalSami\\Deep_Shallow_ml_devel\\'
db_name = 'panda_v1.db'
num_tasks = 2

display_all_tasks = 'Select * from Task'

one_tskst_sql_query = ( 'select TaskSet.Set_ID, Task.*, TaskSet.Successful '
                        'from Task '
                        'inner join TaskSet on Task.Task_ID = TaskSet.TASK1_ID '
                      )

two_tskst_sql_query = ( 'select TaskSet.Set_ID, t1.*, t2.*, TaskSet.Successful '
                        'from Task t1, Task t2 '
                        'inner join TaskSet on t1.Task_ID = TaskSet.TASK1_ID ' 
                        'and '
                        't2.Task_ID = TaskSet.TASK2_ID '
                        'and '
                        'TaskSet.TASK3_ID = -1 and TaskSet.TASK4_ID = -1 '
                     )


three_tskst_sql_query = ('select TaskSet.Set_ID, t1.*, t2.*, t3.*, TaskSet.Successful '
                         'from Task t1, Task t2, Task t3 '
                         'inner join TaskSet on t1.Task_ID = TaskSet.TASK1_ID '
                         'and '
                         't2.Task_ID = TaskSet.TASK2_ID '
                         'and ' 
                         't3.Task_ID = TaskSet.TASK3_ID '
                         'and TaskSet.TASK4_ID = -1 '

                        )

sql_queries = [
    display_all_tasks, one_tskst_sql_query, two_tskst_sql_query, three_tskst_sql_query
]


def read_sql(num_tasksets):
    """
    Forward pass for pytorch model which computes on x (training data) as it is propogated through network.


    Parameters
    ----------
    num_tasksets: int

        Taskset sizes ranging from 1 to 3 (newer databases will support bigger sized tasksets

    Returns
    -------
    df: pandas dataframe
        Data Table of dataset in a nice easy to use/read dataframe
    """

    sql_query = sql_queries[num_tasksets]
    conn = sqlite3.connect(path + 'data\\external\\' + db_name)

    df = pd.read_sql_query(sql_query, conn)

    conn.close()

    logging.info("Loaded data into pandas database ")
    # Raw data for use
    return df


def clean_data (pandas_df, taskset_size):

    # make a copy for safety
    raw_df = pandas_df

    file_name = str(taskset_size) + "_set_data.pkl"

    # COLUMNS TO DROP
    ID = ["Set_ID", "Task_ID"]
    CONSTANT_VALS = ["Deadline", "Quota", "CAPS", "PKG", "CORES", "COREOFFSET", "OFFSET"]

    # pickle.dump(raw_df, open("clean_raw_data.p", "wb"))
    raw_df.to_pickle(path + "data//raw//" + file_name)

    logging.info("Data is successfully cleaned and pickled")
    return raw_df.drop(columns=ID + CONSTANT_VALS)


if __name__=="__main__":
    logging.basicConfig(filename=path + "reports\\ml.log", level=logging.INFO)
    logging.info("Logger started")

    # User enters size of tasksets
    num_args = str(sys.argv[1])

    df = read_sql(int(num_args))

    x, y = build_tensors(clean_data(df, num_args))

    logging.info("Data is ready. Proceed to models")