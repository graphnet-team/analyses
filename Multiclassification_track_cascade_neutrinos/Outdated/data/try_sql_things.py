import sqlite3 as sqlite3

import numpy as np
import pandas as pd
from pandas import cut, read_sql
import pickle as pkl
from random import choices
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimize

indir_db = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"
ourdir_db = "/groups/icecube/peter/storage/Multiclassification/Burn_sample_RD/dev_lvl3_genie_burnsample_v5_RD.db"
indir_list = "/groups/icecube/peter/workspace/analyses/multi_classification_on_stop_and_track_muons/plotting/Comparison_RD_MC/dev_lvl3_genie_burnsample_RD_event_numbers.csv"

import sqlite3

event_no_RD = pd.read_csv(indir_list).reset_index(drop = True)['event_no'].ravel().tolist()[15000000:30000000]

# Connect to the source and destination databases
src_conn = sqlite3.connect(indir_db)
dest_conn = sqlite3.connect(ourdir_db)

# Create a cursor for the source database
src_cursor = src_conn.cursor()

# Select all rows from the source table where the event_no column is in the event_no_MC list
src_cursor.execute("SELECT * FROM truth WHERE event_no IN {}".format(tuple(event_no_RD)))

# Fetch all rows from the SELECT statement
rows = src_cursor.fetchall()

# Execute the PRAGMA table_info command
src_cursor.execute("PRAGMA table_info(truth)")

# Fetch the result set
result = src_cursor.fetchall()

# Extract the column names from the result set
column_names = [(row[1]) for row in result]

# Print the column names
print('({})'.format(', '.join(map(str, column_names))))

# Check if the destination table exists in the destination database
dest_cursor = dest_conn.cursor()
dest_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='truth'")
if not dest_cursor.fetchone():
    # If the destination table does not exist, create it
    dest_cursor.execute("CREATE TABLE truth ({})".format(', '.join(map(str, column_names))))

# Iterate over the rows and insert them into the destination table
for row in rows:
    dest_conn.execute("INSERT INTO truth VALUES ({})".format(','.join(['?'] * len(column_names))), row)

# Commit the transaction to the destination database
dest_conn.commit()

# Close the connections to the databases
src_conn.close()
dest_conn.close()
print('Done')