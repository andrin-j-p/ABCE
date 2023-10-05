import numpy as np
import pandas as pd
import os

# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)

## read data from rawdata file into pandas dataframe
def read_dataframe(file_name):
    # create path to rawdata file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    file = parent_directory + "/data/rawdata/" + file_name

    # open the requested file
    try:
      df = pd.read_stata(file)
      return df
    
    except FileNotFoundError:
      print(" A File not found error occured in open_file.")

    except Exception as e:
      print("An unexpected error occurred in open_file:", e)
    

def data_preprocessing(data):
    pass

def summary_stats():
    pass



