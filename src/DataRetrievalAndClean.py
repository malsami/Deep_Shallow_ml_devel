import sqlite3
import pandas as pd
import torch
# from torch.utils.serialization import load_lua
import pickle
import logging
import sys

# from src.Utils import build_tensors, load_data, db_names
from Utils import build_tensors, load_data, db_names

path = '../'
num_tasks = 1

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


def read_sql(num_tasksets, db):
    """
    Forward pass for pytorch model which computes on x (training data) as it is propogated through network.


    Parameters
    ----------
    num_tasksets: int

        Taskset sizes ranging from 1 to 3 (newer databases will support bigger sized tasksets

    db : string
        The database name that the user decides to use
    Returns
    -------
    df: pandas dataframe
        Data Table of dataset in a nice easy to use/read dataframe
    """

    sql_query = sql_queries[num_tasksets]
    conn = sqlite3.connect(path + 'data/external/' + db)

    df = pd.read_sql(sql_query, conn)


    conn.close()

    logging.info("Loaded data into pandas database ")
    # Raw data for use
    return df


def clean_data (pandas_df, db, taskset_size):

    # make a copy for safety
    raw_df = pandas_df

    file_name = str(taskset_size) + "_set_data.pkl"

    # COLUMNS TO DROP
    ID = ["Set_ID", "Task_ID"]
    CONSTANT_VALS = ["Deadline", "Quota", "CAPS", "PKG", "CORES", "COREOFFSET", "OFFSET"]

    # pickle.dump(raw_df, open("clean_raw_data.p", "wb"))
    raw_df.to_pickle(path + "data/raw/" + db + "_" + file_name)

    logging.info("Data is successfully cleaned and pickled")
    return raw_df.drop(columns=ID + CONSTANT_VALS)


if __name__=="__main__":
    logging.basicConfig(filename=path + "reports//ml.log", level=logging.INFO)
    logging.info("Logger started")

    db_idx = input("Choose DB Version (1 for panda_v1, 2 for panda_v2, ...)\n")
    taskset_size_selection = input("Choose the Taskset size (1,2,3)\n")
    # -1 is for the indexing
    db_name = db_names[int(db_idx) - 1]
    # User enters size of tasksets
    # num_args = sys.argv[1]

    df = read_sql(int(taskset_size_selection), db_name)
    logging.info("DB selected: %s Taskset Size: %s", db_name, taskset_size_selection)
    clean_df = clean_data(df, db_name, taskset_size_selection)
    x, y, _, _ = build_tensors(db_name, clean_df, taskset_size_selection)

    logging.info("Data successfully loaded. Ready for Training.")
    print("Data is ready. Proceed to models")
