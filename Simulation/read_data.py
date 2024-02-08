import numpy as np
import pandas as pd
import os
import json
from shapely.geometry import Point, shape 
import warnings
import geopandas as gpd

# @replace '' and "" consistent

# to supress runtime warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def read_dataframe(file_name, retval="df"):
    """
    Type:         Global helper function 
    Description:  save way to create and return pandas dataframes or relative paths to datafiles
    Used in:      globally
    """
    # create path to rawdata file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    file = parent_directory + "/data/raw_data/" + file_name

    # if the dataframe ought to be returned
    if retval=="df":
    # open the requested file
      try:
        df = pd.read_stata(file)
        return df
      
      except FileNotFoundError:
        print(f"A File not found error occured in open_file when opening {file_name}.")

      except Exception as e:
        print(f"An unexpected error occurred in open_file when opening {file_name}", e)
    
    # if the filename ought to be returned
    elif retval=="file":
      return file
    
    else:
       raise ValueError('The arg "retval" in read_dtaframe is invalid (use df or file)')

# Load GeoJSON file containing sectors
file_path = read_dataframe("filtered_ken.json", retval="file")
with open(file_path) as f:
  js = json.load(f)

def is_in_area(pos_lat, pos_lon):
  """
  Type:         Helper function 
  Description:  Checks if the provided point is within the study area 
  Used in:      read_data.py and Sugarscepe.py 
  """
  point = Point(pos_lon, pos_lat)
  
  # to prevent people from drowning
  if 0.03 <= pos_lat <= 0.08 and 34 <= pos_lon <= 34.18:
    return False

  # check every constituent polygon to see if it contains the point
  for feature in js['features']:
      polygon = shape(feature['geometry'])

      # if the coordinates are within the study area add it to the lists
      if polygon.contains(point):
        county = feature["properties"]["NAME_3"]
        return county
  
  return False


def create_random_coor(N):
  """
  Type:         local helper function to generate N random coordinate in the study area
  Description:  generates N random coordintes in the study area 
  Used in:      Sugarscepe.py to create village and markets at random locations
  """
  # create random coordinates and check if they are within the study area 
  # do this until N coordinates are created
  count = 0
  lon = []
  lat = []
  counties = []
  while count < N:
    # construct point based on lon/lat of the study area
    pos_lat = np.random.uniform(-0.0407, 0.2463)
    pos_lon = np.random.uniform(34.1223, 34.3808)

    # check every constituent polygon to see if it contains the point
    # if the coordinates are within the study area add it to the lists
    county = is_in_area(pos_lat, pos_lon)
    if county != False:
      lon.append(pos_lon)
      lat.append(pos_lat)
      
      # Extract county name from json object and add it to list
      counties.append(county)
      count += 1

  return list(zip(lat, lon)), counties

   
def create_agent_data():
    
    """
    Type:         Function 
    Description:  Does basic data preprocessing and creates a pandas datafrmae based on the Egger dataset 
                  for class instantiation and output validation
    """

### HH Data

    # load household datasets
    df_hh = read_dataframe("GE_HHLevel_ECMA.dta", "df")

    # selcet the variables of interest
    df_hh = df_hh.loc[:, ["hhid_key", "village_code", "p3_totincome", "own_land_acres", "landprice_BL", "landprice", 
                          "treat", "hi_sat", "s12_q1_cerealsamt_12mth", 'p3_totincome_PPP', 'p2_consumption_PPP', 'p4_3_selfemployed']]
    
    # drop 3 agents with negative income and no firm 
    df_hh  = df_hh.drop(df_hh[df_hh['hhid_key'].isin(['601010103003-144', '601020404002-039', '601050304006-038'])].index)
    
### Market Data

    # for markets add randomly created geo-data
    mk_pos, county = create_random_coor(61)
    df_mk = pd.DataFrame({'pos': mk_pos,
                          'county': county,
                          'market_id': np.arange(61)})

### Village Data 

    # Load village data
    df_vl = read_dataframe("GE_VillageLevel_ECMA.dta", "df")

    # add id of closest market to df_vl
    nrst_mk = read_dataframe("Village_NearestMkt_PUBLIC.dta", "df")
    df_vl = pd.merge(df_vl, nrst_mk, on="village_code")

### Firm Data

    # Load firm data
    df_fm = read_dataframe("GE_Enterprise_ECMA.dta", "df")

    # drop enterprises with no matching owner (negligible; see Egger whole apdx)
    df_fm = df_fm[df_fm['hhid_key'] != '']

    # replace missing values with median 
    for col in ['rev_year', 'prof_year', 'owner_education']:
      median = df_fm[col].median()
      df_fm[col].fillna(median, inplace=True) 

    return df_hh, df_fm, df_vl, df_mk


def create_geojson(exectute = False):
  """
  Type:        Helper function
  Description: Creates a filtered verion of the geodata used to display study area boundries
               only needs to be executed once
  Used in:     visualization.py
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
