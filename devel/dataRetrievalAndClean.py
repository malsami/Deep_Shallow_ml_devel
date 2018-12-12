import sqlite3
import pandas as pd
import torch

path = 'C:\\Users\\Varun\\Documents\\Misc\\Research\\MalSami\\'
db_name = 'panda_v1.db'


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

sql_queries = {
    one_tskst_sql_query, two_tskst_sql_query, three_tskst_sql_query
}


def read_SQL(num_Tasksets, db_name = path + db_name):

    sql_query = sql_query[num_Tasksets - 1]
    conn = sqlite3.connect(path + db_name)

    df = pd.read_sql_query(sql_query, conn)

    # Raw data for use
    return df

def clean_data (pandas_df):

    # make a copy for safety
    raw_df = pandas_df

    # COLUMNS TO DROP
    ID = ["Set_ID", "Task_ID"]
    CONSTANT_VALS = ["Deadline", "Quota", "CAPS", "PKG", "CORES", "COREOFFSET", "OFFSET"]

    return raw_df.drop(columns = ID + CONSTANT_VALS)

def build_tensors(clean_df):

    y_tensor = torch.tensor(clean_df['Successful'].values)
    training_val = clean_df.drop('Succesful', axis = 1)
    x_tensor = torch.tensor(training_val.values)

    return (x_tensor, y_tensor)



