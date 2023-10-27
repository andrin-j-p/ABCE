import numpy as np
import pandas as pd
import os
import json
from shapely.geometry import Point, shape 
import warnings
import geopandas as gpd

#@TODO only take columns actually needed in ABM
#      replace '' and "" consistent
#      pickle this

# to supress runtime warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# @DELETE set display options. Not imperative for exectution
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
        print(f"A File not found error occured in open_file when opening {file_name}.")

      except Exception as e:
        print(f"An unexpected error occurred in open_file when opening {file_name}", e)
    
    # if the filename ought to be returned
    elif retval=="file":
      return file
    
    else:
       raise ValueError('The arg "retval" in read_dtaframe is invalid (use df or file)')
    

def create_random_coor(N):
  """
  Type:         local helper function to generate N random coordinate in the study area
  Description:  generates N random coordintes in the study area 
  Used in:      Sugarscepe.py to create village and markets at random locations
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
    Description:  Does basic data preprocessing and creates a pandas datafrmae based on the Egger dataset 
                  for class instantiation and output validation
    """

### HH Data

    # load household datasets
    df_hh = read_dataframe("GE_HHLevel_ECMA.dta", "df")
    df_hh_assets = read_dataframe("GE_HH-BL_assets.dta", "df")

    # selcet the variables of interest
    df_hh = df_hh.loc[:, ["hhid_key", "village_code", "p3_totincome", "own_land_acres", "landprice_BL", "landprice", "treat", "s12_q1_cerealsamt_12mth"]]
    df_hh_assets = df_hh_assets.loc[:, ["hhid_key","h1_1_livestock", "h1_2_agtools", "h1_11_landvalue", "h1_12_loans"]]
    
    # combine all hh data into one df
    df_hh = df_hh.merge(df_hh_assets, on='hhid_key')

    # replace NANs in own_land_acres with zero
    df_hh["own_land_acres"].fillna(0, inplace=True)

    # replace NANs in h1_11_landvalue with an estimate based on land value and land owned
    df_hh['h1_11_landvalue'] = np.where(df_hh['h1_11_landvalue'].isnull(),df_hh['landprice_BL'] * df_hh['own_land_acres'],df_hh['h1_11_landvalue'])    
    
### Market Data
    
    # load market data
    df_mk = read_dataframe("GE_MarketData_Panel_ProductLevel_ECMA.dta", "df")

### Village Data 

    # Load village data
    df_vl = read_dataframe("GE_VillageLevel_ECMA.dta", "df")

    # for villages add randomly created geo-data
    vil_pos, county = create_random_coor(653)
    df_vl["pos"] = (vil_pos)
    df_vl["county"] = county  

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
