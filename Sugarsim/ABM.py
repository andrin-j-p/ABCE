from typing import Any
import mesa
from mesa.model import Model
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import networkx as nx
import random
from read_data import create_agent_data

"""
@TODO preliminary: make this an aggricultural firm later
for now produces an output each time-step 
"""
class Land(mesa.Agent):
  def __init__(self, unique_id, model, size, max_output):
    super().__init__(unique_id, model)
    self.id = unique_id
    self.size = size
    self.max_output = max_output
    self.output = np.random.uniform(1, 10)
  
  def step(self):
     """
     Type:        Method
     Description: Add one unit of output per ha per step until max_output is reached
     """
     # land generates output
     self.output = min([self.max_output, self.output + 1])
    

class Agent(mesa.Agent): 
  """
  Type:         Mesa Agent Class
  Description:  Main entetdy of the simulation 
  """ 
  def __init__(self, unique_id, model, village, income, land_size):
    super().__init__(unique_id, model)
    # initialize geo-related characteristics
    self.village = village
    self.county = self.village.county 
    self.pos = (self.village.pos[0] + np.random.uniform(-0.0001, 0.0001), # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
                self.village.pos[1] + np.random.uniform(-0.0001, 0.0001)) # it is randomly clustered around the village the agent belongs to
    
    # initialize other characteristics
    self.income = income

    # create a land instance for the agent depending on the agents
    self.land = Land(f"l_{self.unique_id}", model, land_size, max_output=500) # pass agent_id as unique_id of the land to identify ownership
    self.model.schedule.add(self.land)

    
  def __repr__(self):
        return f'Agent: {self.unique_id} at coordinates: {self.pos} in county {self.county}'


class Village(mesa.Agent):
    """
    Type:         Mesa Agent Class
    Description:  Phisical location of agents. 
                  level at which spillovers are determined
    """
    def __init__(self, unique_id, model, pos, county, population):
      super().__init__(unique_id, model)
      self.pos = pos
      self.county = county
      self.population = population

    def __repr__(self):
       return f'Village: {self.id} at coordinates {self.pos} in county {self.county}'


class Market(mesa.Agent):
    """
    Type:         Mesa Agent Class
    Description:  Entity were phisical transactions take place
    Used in: 
    """
    def __init__(self, unique_id, model, pos, county, price):
      super().__init__(unique_id, model)
      self.pos = pos
      self.county = county
      self.price = price
      self.people = []

    def __repr__(self):
       return f'Market: {self.id} at coordinates {self.pos} in county {self.county}'


class Sugarscepe(mesa.Model):
  """
  Type:         Mesa Model Class
  Description:  Main Class for the simulation
  """
  def __init__(self, min_lat=-0.05 , max_lat=0.25, min_lon=34.00, max_lon=34.5, N=25):
    # confine the geographic space of the grid to the study area
    self.x_min = min_lat
    self.x_max = max_lat
    self.y_min = min_lon
    self.y_max = max_lon
    self.grid = mesa.space.ContinuousSpace(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, torus=False)

    # initialize exogeneous population attributes
    self.N = N

    # initialize scheduler
    self.schedule = mesa.time.RandomActivationByType(self)

    # add a datacollector 
    self.datacollector = []
    
    # load the dataset used to initialize the village, agent instances and store it 
    self.df_hh, self.df_vl = create_agent_data()

    # create a list of villages based on the loaded df_vl
    village_list = [Village(f"v_{row.id}", self, row.pos, row.county, 0) for row in self.df_vl.itertuples()]

    # create a list of agents based on the loaded df_hh
    agent_list = [Agent(f"a_{row.id}", self, random.choice(village_list), row.p3_totincome, row.own_land_acres ) for row in self.df_hh.itertuples()]
    
    # agentize the grid with N agents
    for agent in agent_list:

          # place agent on grid
          self.grid.place_agent(agent, agent.pos)

          # store grid-location in schedule
          self.schedule.add(agent) 

  def step(self):
    """
    Type:         Method
    Description:  Model step function. Calls all entetie's step functions
    """
    # exectute step for each land entity
    for land in self.schedule.agents_by_type[Land].values():
      land.step()


    self.schedule.steps += 1 # for data collector to track the bumber of steps
    self.collect_data()        
  
  def collect_data(self):
    """
    Type:         Method 
    Description:  Stores all agent-level data generated in a step as a pandas df 
    """
    # collect the agent data for the current step
    step = self.schedule.steps
    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.land.output ) for agent in self.schedule.agents_by_type[Agent].values()]

    # Create a Pandas DataFrame from the comprehension
    df_new = pd.DataFrame(agent_data, columns=['step','id', 'village_id', 'lat', 'lon', "income", "output"])

    # add dataframe of current step to the list 
    self.datacollector.append(df_new)

  def get_data(self):
    """
    Type:        Method
    Description: Concatenate all collected data into a single DataFrame
    """
    # Concatenate the list of dataframes
    stacked_df = pd.concat(self.datacollector, axis=0)

    # Reset the index
    stacked_df.reset_index(drop=True, inplace=True)
    return stacked_df


def run_simulation(steps = 5):
  model = Sugarscepe(N=500)

  for i in range(steps):
     model.step()
  
  return model
