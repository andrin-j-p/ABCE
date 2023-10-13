#%%
import numpy as np
import pandas as pd
import os
import json
from shapely.geometry import Point, shape 
import warnings
import geopandas as gpd

# to supress runtime warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)


def read_dataframe(file_name, retval="df"):
    """
    Type:         Global helper function 
    Description:  save way to create and return pandas dataframes or relative paths to datafiles
    Used in:      globally
    """
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
  Type:         local helper function to generate N random coordinate in the study area
  Description:  generates N random coordintes in the study area 
  Used in:      [class] Sugarscepe to create village and markets at random locations
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
  county = []
  while count < N:
    # construct point based on lon/lat of the study area
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
            county.append(feature["properties"]["NAME_3"])
            count += 1

  return list(zip(lat, lon)), county


def create_agent_data():
    
    """
    Type:         Function 
    Description:  Creates a pandas datafrmae based on the Egger dataset
                  Returns a dataframe with ["id", "lon", "lat", "county"] + [cols_used]
    Used in:      [class] Sugarscape to instantiate agents with the characteristics in the data frame
    """
    # load household datasets
    df_hh = read_dataframe("GE_HHLevel_ECMA.dta", "df")
    df_hh = df_hh.loc[:, ["hhid_key","village_code", "p3_totincome", "own_land_acres", "treat", "s12_q1_cerealsamt_12mth"]]

    #load village data
    df_vl = read_dataframe("GE_VillageLevel_ECMA.dta", "df")

    # load market data
    df_mk = read_dataframe("GE_MarketData_Panel_ProductLevel_ECMA.dta", "df")
     
    # add an index to each dataset and set it as the first column
    for df in [df_hh, df_vl, df_mk]: 
      object_idx = pd.Series(np.arange(0, df.shape[0]))
      df.insert(0, 'id', object_idx)

    # for villages add randomly created geo-data
    vil_pos, county = create_random_coor(653)
    df_vl["pos"] = (vil_pos)
    df_vl["county"] = county  

    # add id of closest market to df_vl
    nrst_mk = read_dataframe("Village_NearestMkt_PUBLIC.dta", "df")
    df_vl = pd.merge(df_vl, nrst_mk, on="village_code")

    
    return df_hh, df_vl, df_mk


def create_geojson(exectute = False):
  """
  Type: Helper function
  Description: creates a filtered verion of the geodata
  Used in: [file] visualization.py
  """
  if exectute == False:
    return

  else:
    # Load the GeoJSON data 
    file_name = read_dataframe("ken.json", retval="file")
    gdf = gpd.read_file(file_name)

    # Extract the three subcounties (Alego, Ugunja and Ukwala) where the experiment took place
    county_list = ['CentralAlego', 'NorthAlego', 'SouthEastAlego', 'WestAlego', 'SiayaTownship', 'Ugunja', 'Ukwala' ]
    filtered_gdf = gdf[gdf['NAME_3'].isin(county_list)]

    # Save the filtered GeoJSON to a new file
    filtered_gdf.to_file('../data/filtered_ken.json', driver='GeoJSON')

  return 
   




# %%
