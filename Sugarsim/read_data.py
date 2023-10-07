#%%
import numpy as np
import pandas as pd
import os
import json
from shapely.geometry import Point, shape 

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
    


def create_random_coor(N):
  """
  Helper function to generate N coordinate in the study area
  used in [class] Sugarscape
  """
  # Load GeoJSON file containing sectors
  file_path = read_dataframe("filtered_ken.json", retval="file")
  with open(file_path) as f:
      js = json.load(f)

  # create random coordinates and check if they are within the study area 
  # do this until N coordinates are created
  count = 0
  lon = []
  lat = []
  features = []
  while count < N:
    # construct point based on lon/lat
    pot_lat = np.random.uniform(-0.0407, 0.2463)
    pot_lon = np.random.uniform(34.1223, 34.3808)
    point = Point(pot_lon, pot_lat)

    # check every constituent polygon to see if it contains the point
    for feature in js['features']:
        polygon = shape(feature['geometry'])

        # if the coordinates are within the study area add it to the lists
        if polygon.contains(point):
            lon.append(pot_lon)
            lat.append(pot_lat)

            # Extract county name from json object and add it to list
            features.append(feature["properties"]["NAME_3"])
            count += 1

  return lon, lat, features


"""
Function to create a datafrmae based on the Egger dataset
Used in model class to instantiate agents with the characteristics in the data frame
Returns a dataframe with ["id", "lon", "lat", "county"] + [cols_used]
"""
def create_agent_data(N = 100, cols_used=[ "p3_totincome", "own_land_acres"]):
    # read the hh data
    df = read_dataframe("GE_HHLevel_ECMA.dta", "df")

    # assert that all colnames are in datafram
    # @TODO fix this
    #assert set(cols_used).issubset(df.columns), "Not all columns are in the DataFrame"
 
    # add an agent index and put it as the first column 
    # @TODO do this once and for all to supress warning i.e. add to frame permanently
    agent_idx = pd.Series(np.arange(0, df.shape[0]))
    df.insert(0, 'id', agent_idx)
    

    # extract the columns of interest for the first N agents
    df = df.loc[:N-1, cols_used]

    # add randomly created geo-data to the hh df
    lon, lat, feature = create_random_coor(N)
    df["lon"] = lon
    df["lat"] = lat
    df["county"] = feature
    return df
  

create_agent_data(100)







