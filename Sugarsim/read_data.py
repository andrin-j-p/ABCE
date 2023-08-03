import numpy as np
import pandas as pd
import os

# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)

def read_dataframe(file_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file = current_directory + "/rawdata/" + file_name

    try:
      df = pd.read_stata(file)
      print(f"read file {file}")
      return df
    
    except FileNotFoundError:
      print(" A File not found error occured in open_file.")

    except Exception as e:
      print("An unexpected error occurred in open_file:", e)
    

def data_preprocessing(data):
    pass

def summary_stats():
    pass


df_location = read_dataframe("CleanGeography_PUBLIC.dta")
