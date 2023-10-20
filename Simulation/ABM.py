import mesa
import numpy as np
import pandas as pd
import matplotlib as mpl
import networkx as nx
from read_data import create_agent_data
import timeit
import cProfile # move to Test.py
import pstats   # move to Test.py
np.random.seed(1)

# @TODO 
# set a global seed
# add mock class to testsuit

class Firm(mesa.Agent):
  """
  Type:        Mesa Agent Class
  Description: Implements firm entity
  """
  def __init__(self, unique_id, model, owner, market):
    super().__init__(unique_id, model)
    # firm properties
    self.id = unique_id
    self.owner = owner
    self.market = market
    # price
    self.price = np.random.uniform(1, 10) # price in current month
    self.marginal_cost = 1 # @Calibrate @set to wage
    self.max_price = 100 #@ set to reasonable value
    self.theta = 0.5 # probability for price change
    self.nu = 0.3 # rate for price change @Calibrate. 
    # wage
    self.wage = 1 # @TODO depending on productivity
    self.etha = 0.1 # rate for wage change 
    self.n_employees = 10
    self.employees = [1,1,1] # @
    # inventory 
    self.min_stock = 0 # 10% percent of last months sales @TODO perishable?
    self.max_stock = 0 # 150% percent of last months sales 
    self.stock = 6
    self.output = 0 # quantity produced this month
    self.sales = 0  # quantity sold this month
    # profit
    self.productivity = np.random.uniform(0, 2) # @Calibrate
    self.money = 0 #@ refernce for calibration

  def set_price(self):
    """
    Type:        Method 
    Description: Implements heuristics for price setting
    Exectuted:   Monthly 
    """
    current_price = self.price
    # price is increased with probability theta if the follwoing conditions are met:
    # 1) the stock is below the minimum inventory
    # 2) price last month was strictly smaller than the maximum price
    if self.stock < self.min_stock and current_price * (1 + self.nu) < self.max_price:
      if np.random.uniform(0, 1) < self.theta:
        current_price *= 1 + self.nu

    # price is decresed with probability theta if the follwoing conditions are met:
    # 1) the stock exceeds the maximum inventory
    # 2) price last month was strictly larger than the marginal cost
    elif self.stock > self.max_stock and current_price * (1-self.nu) > self.marginal_cost:
      if np.random.uniform(0, 1) < self.theta:
        current_price *= 1 - self.nu

    # update price
    self.price = current_price

  def produce(self):
    """
    Type:        Method 
    Description: Generate output according to firm idiosyncratic production function @ source of heterog
    Execute:     Monthly 
    """
    # output is a function of number of employees and productivity parameter
    amount = self.productivity * len(self.employees)
    self.output += amount
    self.stock += amount
    self.money -= amount * self.marginal_cost
    
  def set_labor(self):
    """
    Type:        Method 
    Description: The amount of labor and thereby the level of production is chosen
    Execute:     Monthly ? (leninik hiring is performed daily)
    """
    # a worker is hired if the inventory at the end of the month is below the minimum stock 
    if self.stock < self.min_stock:
      self.employees.append(1)

    # the least productive worker is fired if the following two conditions are satisfied
    # 1) there are workers
    # 2) the inventory at the end of the month is above the maximum stock
    elif self.stock > self.max_stock and len(self.employees) > 0:
      self.employees.pop()

  def distribute_profit(self):
    """
    Type:        Method 
    Description: Pay wages and distribute the remaining profits
    Execute:     Monthly 
    """
    # pay wages based on the employees productivity 
    for employee in self.employees:
      # productivity is observed imperfectly. Wages are thus fluctuating 10% above and below actual productivity
      prod = employee.porductivity
      employee.money += prod + np.random.uniform(prod -(prod * 10 /100), prod + (prod *10/100))
    pass

  def step(self):
    """
    Type:        Method
    Description: Firm step function 
    """
    # if it is the end of a month the month
    if self.model.schedule.steps%5 == 0:
      # set the min_stock to the sales of previous month
      self.min_stock = self.sales * 0.1
      self.max_stock = self.sales * 10
      # reset sales and output for this month to zero
      self.sales = 0
      self.output = 0
      # produce output for this month
      #self.set_wage()
      self.set_labor()
      self.produce()
      # set price for this month
      self.set_price()


class Agent(mesa.Agent):
  """
  Type:         Mesa Agent Class
  Description:  Represents a household in the economy
  """
  def __init__(self, unique_id, model, village, income, Firm):
    super().__init__(unique_id, model)
    # initialize geo-related characteristics
    self.village = village
    self.county = self.village.county
    self.pos = (self.village.pos[0] + np.random.uniform(-0.0003, 0.0003), # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
                self.village.pos[1] + np.random.uniform(-0.0003, 0.0003)) # it is randomly clustered around the village the agent belongs to
    # initialize other hh characteristics
    self.income = income
    self.porductivity = 1
    self.firm = Firm 
    self.employer = None
    self.best_dealers = []
    # initialize consumption related characteristics
    # @TODO change this to four categories:Food,Livestock, Non-Food Non-Durables, Durables, Temptation Goods
    #       create an index for all four cats

    # @TODO Preliminary. for each of the following vars create algo on how to be caluculated
    #       distinguish between perishable and non-perishable good
    # initialize trade variables
    self.market_day = np.random.randint(0, 7) # day the agent goes to market
    self.best_dealer_price = 1000 # agent remembers price of best dealer last week
    self.money = 100
    self.demand = 1
    self.consumption = 0

  def find_dealer(self):
    """
    Type:         Method 
    Description:  Maintains the list of best dealers  
    Used in:      buy_goods
    """
    # retrieve the list of all firms operating on the market
    potential_dealers = self.village.market.vendors

    # market needs at least three dealer on a given day to work
    if len(potential_dealers) < 4:
      return False
    
    # if the list of best dealers is empty try 3 'random dealers'
    if len(self.best_dealers) == 0:
        self.best_dealers = list(np.random.choice(potential_dealers, size=3, replace=False))

    # if price of best dealer increased compared to last week try a new dealer at random @spite
    # only checked from the second week onwards when the self.best_dealer_price contains a value
    elif self.best_dealers[0].price > self.best_dealer_price:

      # randomly select a new dealer from the dealers not already in the list and append it at the end 
      # note: np.random.choice returns a np. array; thus the index
      new_dealer = np.random.choice(list(set(potential_dealers).intersection(set(self.best_dealers))), size=1, replace=False)[0]
      self.best_dealers[-1] = new_dealer

    # sort the list of dealers according to the price they offer
    self.best_dealers = sorted(self.best_dealers, key=lambda x: x.price)
    self.best_dealer_price = self.best_dealers[0].price

    # return the first dealer in the list which has enough of the good on stock
    for dealer in self.best_dealers:
      if dealer.stock > self.demand:
        return dealer
      
    return False

  def trade(self, dealer):
    """
    @TODO implement partially satisfied demand see Lengnickp 109
    Type:        Method
    Description: Exchange of goods
    """
    # calculate the trade volume
    price = dealer.price
    total_price =  price * self.demand
    
    # change the affected demand side variables
    self.money = self.money - total_price
    self.consumption + self.demand

    # change the affecteed supply side variables
    dealer.stock -= self.demand
    dealer.sales += self.demand
    dealer.money += total_price 
    
    # save the transaction details in the model data collector
    self.model.m_datacollector.append({"step": self.model.schedule.steps, "parties": (self.unique_id, dealer.owner), 
                                       "price": price, "amount": self.demand, "total_price": total_price})

  def step(self):
    # hh step only needs to be executed on market day
    if self.market_day == self.model.schedule.steps%5:
      dealer = self.find_dealer()
      # If there is an available dealer trade
      if dealer:
        self.trade(dealer)

  def __repr__(self):
    return f'Agent: {self.unique_id} at coordinates: {self.pos} in county {self.county}'


class Village(mesa.Agent):
    """
    Type:         Mesa Agent Class
    Description:  Physical location of households. Deterimines social environment of agents
                  level at which spillovers occur
    """
    def __init__(self, unique_id, model, pos, county, market):
      super().__init__(unique_id, model)
      self.pos = pos
      self.county = county
      self.market = market
      self.population = None

    def __repr__(self):
       return f'Village: {self.unique_id} at coordinates {self.pos} in county {self.county}'


class Market(mesa.Agent):
  """
  Type:         Mesa Agent Class
  Description:  Entity were phisical transactions take place. 
  Used in:      Agent.trade(), 
  """
  def __init__(self, unique_id, model, data):
    super().__init__(unique_id, model)
    self.data = data
    self.vendors = None
    

  def step(self):
     # reset the list of current costumers back to 0
     pass
  
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

    # add an agent, firm and model datacollector
    self.a_datacollector = []
    self.f_datacollector = []
    self.m_datacollector = []

    # load the dataset used to initialize the village, agent instances and store it
    self.df_hh, self.df_fm, self.df_vl, self.df_mk = create_agent_data()

    # create a dictionary of markets based on the loaded df_mk
    # note: the self parameter passed explicitly is the model i.e. the current instance of Sugarscape
    mk_dct = {market_id: Market(unique_id=f"m_{market_id}", model=self, data=self.df_mk.loc[self.df_mk['market_id'] == market_id])
              for market_id in self.df_mk["market_id"].unique()}

    # create a dictionary of villages based on the loaded df_vl
    vl_dct = {row.village_code : Village(unique_id=row.village_code, model=self, pos=row.pos, county=row.county, 
                                         market=mk_dct[row.market_id])
              for row in self.df_vl.itertuples()}
    
    # create a dictionary of firms based on loaded df_fm 
    fm_dct = {row.hhid_key: Firm(unique_id=f"f_{row.hhid_key}", model=self, owner = row.hhid_key, market=vl_dct[row.village_code].market)
              for row in self.df_fm.itertuples()}

    # create a list of agents based on the loaded df_hh
    hh_lst = [Agent(unique_id=row.hhid_key, model=self, village=vl_dct[row.village_code], income=row.p3_totincome,
                    Firm=fm_dct.get(row.hhid_key, None))
              for row in self.df_hh.itertuples()]

    # agentize the grid with hhs
    for hh in hh_lst:
      # place agent on grid
      self.grid.place_agent(hh, hh.pos)
      # add hh to schedule
      self.schedule.add(hh)
      # add firms to schedule if applicable
      # note: done here so only firms with owners in the GE_HHLevel_ECMA.dta dataset are added
      if hh.firm != None:
        self.schedule.add(hh.firm)
    
    # create an attribute for quick acess to all hh in the model
    self.all_agents = self.schedule.agents_by_type[Agent].values()
  
    # create an attribute for quick acess to all firms in the model 
    self.all_firms = self.schedule.agents_by_type[Firm].values()  

    # agentize the grid with mks
    for mk in mk_dct.values():
       # add mk to schedule
       self.schedule.add(mk)
       # initialize vendors
       mk.vendors = [firm for firm in self.all_firms if firm.market == mk] 

    # create an attribute for quick acess to all firms in the model 
    self.all_markets = self.schedule.agents_by_type[Market].values()  

    # agentize the grid with vls
    for vl in vl_dct.values():
       # add vl to schedule
       self.schedule.add(vl)
       # initialize population 
       vl.population = [hh for hh in self.all_agents if hh.village == vl]
    
    # create an attribute for quick acess to all firms in the model 
    self.all_villages = self.schedule.agents_by_type[Village].values()  

  def randomize_agents(self, agent_type):
    """
    Type:        Helper Method
    Description: Used to create a ranomized list of all hh, fm, mk in the model
                 Required to avoid first mover advantages (see f.i. Axtell 2022 p. ?)
    """
    a = f"all_{agent_type}"
    agent_list = getattr(self, a)
    agent_shuffle = list(agent_list)
    return agent_shuffle

  def step(self):
    """
    Type:         Method
    Description:  Model step function. Calls all entities' step functions
    """
    # exectute step for each firm entity
    fm_shuffle = self.randomize_agents('firms')
    for firm in fm_shuffle:
      firm.step()

    # create randomized list of hh and call their step functions 
    hh_shuffle = self.randomize_agents('agents')
    for hh in hh_shuffle:
        hh.step() # executes 'find_dealer()' and 'trade()'

    # create ranomized list of mk and call their step function
    mk_shuffle = self.randomize_agents('markets')
    for mk in mk_shuffle:
      mk.step() # resets costumers list to []

    self.schedule.steps += 1 # for data collector to track the number of steps
    self.collect_data()

  def collect_data(self):
    """
    Type:         Method
    Description:  Stores all agent-level data generated in a step as a pandas df
    """
    # collect hh and firm data for the current step
    step = self.schedule.steps
    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.best_dealers,
                   agent.consumption, agent.money)
                  for agent in self.all_agents]
    
    firm_data = [(step, firm.unique_id, firm.stock, firm.price, firm.money, firm.output, firm.sales, firm.price * firm.sales,
                  len(firm.employees))
                 for firm in self.all_firms]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_hh_new = pd.DataFrame(agent_data, columns=['step','unique_id', 'village_id', 'lat', 'lon', "income", "best_dealers", "consumption", "money"])
    df_fm_new = pd.DataFrame(firm_data, columns=['step', 'unique_id', 'stock', 'price', 'money', 'output', 'sales', 'revenue', 'employees'])
 
    # add dataframe of current step to the list
    self.a_datacollector.append(df_hh_new)
    self.f_datacollector.append(df_fm_new)
    
  def get_data(self):
    """
    Type:        Method
    Description: Concatenates all collected data into DataFrames for hh, fm and md
    """
    # Concatenate the list of dataframes
    hh_df_stacked = pd.concat(self.a_datacollector, axis=0)
    fm_df_stacked = pd.concat(self.f_datacollector, axis=0)
    md_df_stacked = pd.DataFrame(self.m_datacollector)

    # Reset the index 
    #@NEEDED?
    hh_df_stacked.reset_index(drop=True, inplace=True)
    return hh_df_stacked, fm_df_stacked, md_df_stacked
  
  def run_simulation(self, steps= 25):
    for i in range(steps):
      print(f"\rSimulating step: {i + 1} ({round((i + 1)*100/steps, 0)}% complete)", end="", flush=True)
      self.step()
    print("\n")

def run_simulation(steps = 100):
  start = timeit.default_timer()

  model = Sugarscepe()
  model.run_simulation(steps)
  hh_data, fm_data, md_data = model.get_data()
  print(fm_data[fm_data["step"]==24].head(100))
  
  for i in range(steps):
    print(md_data[md_data['step']== i]['price'].mean())

  stop = timeit.default_timer()
  print('Time: ', stop - start)  
  return model

def batch_runner(steps = 2):
   pass

if __name__ == "__main__":
    cProfile.run("run_simulation()", filename="../data/profile_output.txt", sort='cumulative')
    
    # Create a pstats.Stats object from the profile file
    profile_stats = pstats.Stats("../data/profile_output.txt")

    # Sort and print the top N entries with the highest cumulative time
    #profile_stats.strip_dirs().sort_stats('cumulative').print_stats(20)