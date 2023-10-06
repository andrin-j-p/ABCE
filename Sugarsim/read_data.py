#%%
import numpy as np
import pandas as pd
import os

# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)


"""
Global helper function to read pands dataframes or return relpaths to datafiles
"""
def read_dataframe(file_name, retval="df"):
    # create path to rawdata file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    file = parent_directory + "/data/" + file_name

    # if the dataframe ought to be returned
    if retval=="df":
    # open the requested file
      try:
        df = pd.read_stata(file)
        return df
      
      except FileNotFoundError:
        print(" A File not found error occured in open_file.")

      except Exception as e:
        print("An unexpected error occurred in open_file:", e)
    
    # if the filename ought to be returned
    elif retval=="file":
      return file
    
    else:
       raise ValueError('The arg "retval" in read_dtaframe is invalid (use df or file)')
    

def data_preprocessing(data):
    pass

# %%
def summary_stats(df):
    df.describe()

<<<<<<< HEAD
df_location = read_dataframe("CleanGeography_PUBLIC.dta")
df_hh1 = read_dataframe("GE_HH-Census-BL_PUBLIC.dta")
df_hh2 = read_dataframe("GE_HH-Survey-BL_PUBLIC.dta")
df_hh3 = read_dataframe("GE_HH-SampleMaster_PUBLIC.dta")
#summary_stats(df_hh1)
=======
#df_location = read_dataframe("CleanGeography_PUBLIC.dta")
#df_hh1 = read_dataframe("GE_HH-Census-BL_PUBLIC.dta")
#df_hh2 = read_dataframe("GE_HH-Survey-BL_PUBLIC.dta")
#df_hh3 = read_dataframe("GE_HH-SampleMaster_PUBLIC.dta")

>>>>>>> branch3




# %%
