from typing import Any
import mesa
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import json
from read_data import read_dataframe
from shapely.geometry import Point, shape 


def is_in_kenya(pot_lat, pot_lon):
  """
  Helper function to identify if a given coordinate is in Kenya
  used in [class] Sugarscape
  """
  # Load GeoJSON file containing sectors
  file_path = read_dataframe("filtered_ken.json", retval="file")
  with open(file_path) as f:
      js = json.load(f)

  # construct point based on lon/lat
  point = Point(pot_lon, pot_lat)

  # check every constituent polygon to see if it contains the point
  for feature in js['features']:
      polygon = shape(feature['geometry'])
      if polygon.contains(point):
          #print('Found containing polygon:', feature)
          return True, feature

  # return false if it is not part of any polygon
  return False, None


class Agent(mesa.Agent): 
  """
  Agent: 
  - has a metabolism
  
  """
  def __init__(self, unique_id, model, pos, geo_feature):
    super().__init__(unique_id, model)
    self.id  = unique_id
    self.pos = pos
    self.geo_feature = geo_feature # json object

  def __repr__(self):
        return f'Agent: {self.id} at coordinates: {self.pos} constituency {self.geo_feature["properties"]["NAME_3"]}'


class Sugarscepe(mesa.Model):

  def __init__(self, min_lat=-4.68 , max_lat=4.67, min_lon=33.91, max_lon=41.89, N=5):
    # set the min_x to min lat in Kenya and max_x to max_lat
    self.x_min = min_lat
    self.x_max = max_lat
    self.y_min = min_lon
    self.y_max = max_lon
    self.grid = mesa.space.ContinuousSpace(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, torus=False)
    self.N = N

    #initialize scheduler
    self.schedule = mesa.time.RandomActivation(self)

    # Agentize the grid with N agents
    agent_id = 0
    while agent_id < N:
       # randomly create coordinates and check if they are in Kenya. 
       # The random corrdinates are within a rectangular grid around kenya
       lat = np.random.uniform(-0.0407, 0.2463 )
       lon = np.random.uniform(34.1223, 34.3808 )
       res, geo_feature = is_in_kenya(lat, lon)

       # if the coordinates are inside Kenya create- and place an agent there
       if res == True:
          agent = Agent(agent_id, self, (lat, lon), geo_feature)
          print(agent)

          # place agent on grid
          self.grid.place_agent(agent, (lat, lon))

          # store grid-location in schedule
          self.schedule.add(agent) 

          # update agent id
          agent_id += 1

  def get_agent_pos(self):
    """
    Function to return coordinates of all agents in model
    """
    return [agent.pos for agent in self.schedule.agents]


def run_simulation():
  model = Sugarscepe()
  return model.get_agent_pos()

