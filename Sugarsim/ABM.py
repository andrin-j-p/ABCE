from typing import Any
import mesa
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import networkx as nx
import random
from read_data import create_agent_data


class Agent(mesa.Agent): 
  """
  Agent: 
  - has a metabolism
  
  """
  def __init__(self, unique_id, model, village, income, land):
    super().__init__(unique_id, model)
    self.id  = unique_id
    self.village = village
    self.county = self.village.county 

    # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
    # it is randomly clustered around the village the agent belongs to
    self.pos = (self.village.pos[0] + np.random.uniform(-0.0001, 0.0001), self.village.pos[1] + np.random.uniform(-0.0001, 0.0001))
    self.income = income
    self.land = land

  def __repr__(self):
        return f'Agent: {self.id} at coordinates: {self.pos} in county {self.county}'


class Village(mesa.Agent):
    def __init__(self, unique_id, model, pos, county, population):
      super().__init__(unique_id, model)
      self.pos = pos
      self.county = county
      self.id  = unique_id
      self.population = population

    def __repr__(self) -> str:
       return f'Villag: {self.id} at coordinates {self.pos} in county {self.county}'



class Sugarscepe(mesa.Model):

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
    self.schedule = mesa.time.RandomActivation(self)

    # load the dataset used to initialize the village, agent instances and store it 
    self.df_hh, self.df_vl = create_agent_data()

    # create a list of villages 
    village_list = [Village(row.id, self, row.pos, row.county, 0) for row in self.df_vl.itertuples()]

    # create a list of agents based on the loaded df 
    agent_list = [Agent(row.id, self, random.choice(village_list), row.p3_totincome, row.own_land_acres ) for row in self.df_hh.itertuples()]
    
    # agentize the grid with N agents
    for agent in agent_list:

          # place agent on grid
          self.grid.place_agent(agent, agent.pos)

          # store grid-location in schedule
          self.schedule.add(agent) 
    print("I was called")


# @TODO generalize this
  def get_data(self, object_type="hh"):
    """
    Function to return the atribtues of all agents as df 
    -coordinates
    -income
    -age
    -county where positioned
    """
    # make this one line
    agent_data = [(agent.id, agent.village.id, agent.pos[0], agent.pos[1], agent.income, agent.land ) for agent in self.schedule.agents]

    # Create a Pandas DataFrame from the comprehension
    df = pd.DataFrame(agent_data, columns=['id', 'village_id', 'lat', 'lon', "income", "land"])
    return df


def run_simulation():
  model = Sugarscepe(N=500)
  model.get_data()
  
  return model