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
    one_tskst_sql_query, two_tskst_sql_query, three_tskst_sql_query
]


def read_sql(num_Tasksets):

    sql_query = sql_queries[num_Tasksets - 1]
    conn = sqlite3.connect(path + "data\\raw\\" + db_name)

    df = pd.read_sql_query(sql_query, conn)

    conn.close()

    logging.info("Loaded data into pandas database ")
    # Raw data for use
    return df


def clean_data (pandas_df):

    # make a copy for safety
    raw_df = pandas_df

    # COLUMNS TO DROP
    ID = ["Set_ID", "Task_ID"]
    CONSTANT_VALS = ["Deadline", "Quota", "CAPS", "PKG", "CORES", "COREOFFSET", "OFFSET"]

    # pickle.dump(raw_df, open("clean_raw_data.p", "wb"))
    raw_df.to_pickle("./clean_raw.pkl")

    logging.info("Data is successfully cleaned and pickled")
    return raw_df.drop(columns=ID + CONSTANT_VALS)


if __name__=="__main__":
    logging.basicConfig(filename=path + "reports\\ml.log", level=logging.INFO)
    logging.info("Logger started")

    num_args = str(sys.argv[1])

    df = read_sql(int(num_args))

    x, y = build_tensors(clean_data(df))

    logging.info("Data is ready. Proceed to models")