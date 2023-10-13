import mesa
import numpy as np
import pandas as pd
import matplotlib as mpl
import networkx as nx
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
    # @TOOD preliminary 
    # make this a price vector containing the prices at each time step
    # vector is used for learning and for costumers to detect price changes
    self.price_vec = []

  def set_price(self):
    """
    @TODO implement the learning algo
    Type: Method 
    Description: sets an optimal price using Reinforcment learning  
    """
    price = np.random.randint(0,10)
    self.price_vec.append(price) 


  def step(self):
     """
     Type:        Method
     Description: Add one unit of output per ha per step until max_output is reached
     """
     # land generates output
     self.output = min([self.max_output, self.output + 1])
     self.set_price()


class Agent(mesa.Agent):
  """
  Type:         Mesa Agent Class
  Description:  Main entetdy of the simulation
  """
  def __init__(self, unique_id, model, village, income, land_size, food_cons):
    super().__init__(unique_id, model)
    # initialize geo-related characteristics
    self.village = village
    self.county = self.village.county
    self.pos = (self.village.pos[0] + np.random.uniform(-0.0001, 0.0001), # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
                self.village.pos[1] + np.random.uniform(-0.0001, 0.0001)) # it is randomly clustered around the village the agent belongs to

    # initialize other characteristics
    self.income = income
    self.land = Land(f"l_{self.unique_id}", model, land_size, max_output=500) # create a land instance for the agent pass agent_id to identify ownership
    self.best_dealers = []
    # initialize consumption related characteristics
    # @TODO change this to four categories:Food,Livestock, Non-Food Non-Durables, Durables, Temptation Goods
    #       create an index for all four cats
    #       make it 12 months
    # here just cereals last 12 months(part of food )
    self.food_cons = food_cons
    self.market_day = np.random.randint(0, 7) # day the agent goes to market

  def buy_goods(self, amount):
    """
    Type:         Method  
    Description:  Buy 
    """
    # market needs at least three costumers on a given day to work
    if len(self.village.market.costumers) < 3:
      return
    
    
    # if the list of best dealers is empty try 3 'random dealers'
    if len(self.best_dealers) == 0:
        self.best_dealers = list(np.random.choice(self.village.market.costumers, size=3, replace=False))

    # if price of best dealer increased compared to last week try a new dealer at random 
    # only checked from the second week onwards when the price vector contains two elements 
    elif self.best_dealers[0].land.price_vec[0] > self.best_dealers[0].land.price_vec[1]:
      # randomly select a new dealer and append it at the end (np.random.choice returns a np. array)
      new_dealer = np.random.choice(self.village.market.costumers, size=1, replace=False)[0]
      self.best_dealers[-1] = new_dealer

    # @TODO change this once Firm is implemented
    # sort the list of dealers according to the price they offer
    self.best_dealers = sorted(self.best_dealers, key=lambda x: x.land.price_vec[0])

  def step(self):
    self.buy_goods(10)

  def __repr__(self):
    return f'Agent: {self.unique_id} at coordinates: {self.pos} in county {self.county}'


class Village(mesa.Agent):
    """
    Type:         Mesa Agent Class
    Description:  Physical location of agents. Deterimines social environment of agents
                  level at which spillovers are determined
    """
    def __init__(self, unique_id, model, pos, county, population, market):
      super().__init__(unique_id, model)
      self.pos = pos
      self.county = county
      self.population = population
      self.market = market

    def __repr__(self):
       return f'Village: {self.unique_id} at coordinates {self.pos} in county {self.county}'


class Market(mesa.Agent):
  """
  Type:         Mesa Agent Class
  Description:  Entity were phisical transactions take place
  Used in:
  """
  def __init__(self, unique_id, model, price, data):
    super().__init__(unique_id, model)
    self.price = price
    self.data = data
    self.costumers = []

  def step(self):
     # reset the list of current costumers back to 0
     self.costumers = []
  
  def __repr__(self):
     return f'Market: {self.unique_id} with costumers id: {[costumer.id for costumer in self.costumers]}'


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
    self.df_hh, self.df_vl, self.df_mk = create_agent_data()

    # create a dictionary of villages based on the loaded df_vl
    mk_dct = {market_id: Market(f"m_{market_id}", self, 0, self.df_mk.loc[self.df_mk['market_id'] == market_id])
              for market_id in self.df_mk["market_id"].unique()}

    # create a dictionary of villages based on the loaded df_vl
    vl_dct = {row.village_code : Village(f"v_{row.id}", self, row.pos, row.county, 0, mk_dct[row.market_id])
              for row in self.df_vl.itertuples()}

    # create a list of agents based on the loaded df_hh
    hh_lst = [Agent(row.hhid_key, self, vl_dct[row.village_code], row.p3_totincome, row.own_land_acres,
              row.s12_q1_cerealsamt_12mth)
              for row in self.df_hh.itertuples()]

    # agentize the grid with hhs
    for hh in hh_lst:
      # place agent on grid
      self.grid.place_agent(hh, hh.pos)

      # add hh and its land to schedule
      self.schedule.add(hh)
      self.schedule.add(hh.land) # created when agent is initialized

    for mk in mk_dct.values():
       # add mk to schedule
       self.schedule.add(mk)

    for vl in vl_dct.values():
       # add vl to schedule
       self.schedule.add(vl)

  def randomize_agents(self, agent_type):
    """
    Type:        Helper Method
    Description: Used to create a ranomized list of all hh in the model
                 Important to avoid first mover advantages (see f.i. Axtell 2022 p. ?)
    Used in:
     """
    agent_shuffle = list(self.schedule.agents_by_type[agent_type].values())
    self.random.shuffle(agent_shuffle)
    return agent_shuffle

  def step(self):
    """
    Type:         Method
    Description:  Model step function. Calls all entetie's step functions
    """
    # exectute step for each land entity
    for land in self.schedule.agents_by_type[Land].values():
      land.step()

    # create randomized list of hh and call their step function 
    hh_shuffle = self.randomize_agents(Agent)
    for hh in hh_shuffle:

      # move agent to market if it its market day
      # all agents need to be moved to the market before their buy functions are called in step() 
      if hh.market_day == self.schedule.steps%6:
        hh.village.market.costumers.append(hh)
        hh.step()

    # create ranomized list of mk and call their step function
    mk_shuffle = self.randomize_agents(Market)
    for mk in mk_shuffle:
      mk.step()

    self.schedule.steps += 1 # for data collector to track the number of steps
    self.collect_data()

  def collect_data(self):
    """
    Type:         Method
    Description:  Stores all agent-level data generated in a step as a pandas df
    """
    # collect the agent data for the current step
    step = self.schedule.steps
    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.land.output,
                   agent.best_dealers)
                  for agent in self.schedule.agents_by_type[Agent].values()]

    # Create a Pandas DataFrame from the comprehension
    df_new = pd.DataFrame(agent_data, columns=['step','unique_id', 'village_id', 'lat', 'lon', "income", "output", "best_dealers"])

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


def run_simulation(steps = 14):
  model = Sugarscepe(N=500)

  for i in range(steps):
     print(f"#######\nStep {i}\n#######")
     model.step()
  #print(model.get_data().head())
  return model

def batch_runner(steps = 5):
   pass

#run_simulation()