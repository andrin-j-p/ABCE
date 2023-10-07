from typing import Any
import mesa
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import networkx as nx
from read_data import create_agent_data


class Agent(mesa.Agent): 
  """
  Agent: 
  - has a metabolism
  
  """
  def __init__(self, unique_id, model, pos, county, income, land):
    super().__init__(unique_id, model)
    self.id  = unique_id
    self.pos = pos # tuple of shape (lat, lon) as used for continuous_grid in mesa
    self.county = county 
    self.income = income
    self.land = land

  def __repr__(self):
        return f'Agent: {self.id} at coordinates: {self.pos} constituency {self.county}'


class Sugarscepe(mesa.Model):

  def __init__(self, min_lat=-4.68 , max_lat=4.67, min_lon=33.91, max_lon=41.89, N=25):
    # set the min_x to min lat in Kenya and max_x to max_lat
    self.x_min = min_lat
    self.x_max = max_lat
    self.y_min = min_lon
    self.y_max = max_lon
    self.grid = mesa.space.ContinuousSpace(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, torus=False)
    self.N = N

    #initialize scheduler
    self.schedule = mesa.time.RandomActivation(self)

    # load the dataset containing the columns speciefied and store it as a model attribute
    cols_used = ["id", "p3_totincome", "own_land_acres"]
    self.df = create_agent_data(self.N, cols_used=cols_used)
    
    print("I have been called")
    # create a list of agents based on the loaded df 
    agent_list = [Agent(row.id, self, (row.lat, row.lon), row.county, row.p3_totincome, row.own_land_acres ) for row in self.df.itertuples()]
    
    # Agentize the grid with N agents
    for agent in agent_list:

          # place agent on grid
          self.grid.place_agent(agent, agent.pos)

          # store grid-location in schedule
          self.schedule.add(agent) 


  def get_agent_info(self):
    """
    Function to return the atribtues of all agents as df 
    -coordinates
    -income
    -age
    -county where positioned
    """
    pass


def run_simulation():
  model = Sugarscepe(N=5000)
  return model

