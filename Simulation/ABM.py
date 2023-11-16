import mesa
import numpy as np
import pandas as pd
import matplotlib as mpl
import networkx as nx
from read_data import create_agent_data
from data_collector import Datacollector
import timeit
import copy

import cProfile # move to Test.py
import pstats   # move to Test.py


# @TODO 
# implement cobb douglas? demand
# implement cobb douglas? production function 
# handle negative demand caused by 0.7 * income

class Firm(mesa.Agent):
  """
  Type:        Mesa Agent Class
  Description: Implements firm entity
  """
  # @Concept to reduce variable amount!!! 
  def __init__(self, unique_id, model, market, village):
    super().__init__(unique_id, model)
    # calibraiton variables
    self.mu = 0.1
    self.sigma = 0.4
    # firm properties
    self.id = unique_id
    self.owner = None
    self.market = market
    self.village = village
    self.productivity = 2 #float(np.random.lognormal(self.mu, self.sigma, size=1)/4 + 1) #np.random.uniform(1.5, 2.5) # @Calibrate: main variable. To tune owner income i.e. self.money
    # price
    self.price = np.random.uniform(1, 10) # price in current month (initialized randomly)
    self.marginal_cost = 1 # @CALIBRATE: if necessary
    self.max_price = 11 #@ set to reasonable value
    self.theta = 0.8 # probability for price change
    self.nu = 0.5 # rate for price change @Calibrate: if necessary 
    self.phi_l = 0.1 # phi_l * sales = minimal stock
    self.phi_u = 1 # phi_u * sales = max stock for 
    self.employees = [] 
    # inventory 
    self.min_stock = 0 # 10% percent of last months sales @TODO perishable?
    self.max_stock = 0 # 150% percent of last months sales 
    self.stock = 300
    self.output = 0 # quantity produced this month
    self.sales = 0  # quantity sold this month
    # profit
    self.money = 0
    self.assets = 0
    self.profit = 0
    self.revenue = 0

    #@DELETE
    self.costumers = []

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
      if np.random.uniform(0, 1) <= self.theta:
        current_price *= 1 + self.nu

    # price is decresed with probability theta if the follwoing conditions are met:
    # 1) the stock exceeds the maximum inventory
    # 2) price last month was strictly larger than the marginal cost
    elif self.stock > self.max_stock and current_price * (1-self.nu) > self.marginal_cost:
      if np.random.uniform(0, 1) <= self.theta:
        current_price *= 1 - self.nu

    # update price
    self.price = current_price

  def produce(self):
    """
    Type:        Method 
    Description: Generate output according to firm idiosyncratic production function @ source of heterog
    Execute:     Monthly 
    """
    # Output is a function of employees' and firms' productivity parameters
    amount = self.productivity * (sum(employee.productivity for employee in self.employees)  + self.owner.productivity)
    self.output += amount
    self.stock += amount
    
  def set_labor(self):
    """
    Type:        Method 
    Description: The amount of labor and thereby the level of production is chosen
    Execute:     Monthly ? (leninik hiring is performed daily)
    """
    # A random worker is hired if:
    # 1) the inventory at the end of the month is below the minimum stock 
    # 2) if there is a worker available
    if self.stock < self.min_stock:
      try:
        new_employee = np.random.choice([hh for hh in self.village.population if hh.employer == None], size=1, replace=False)[0]
        new_employee.employer = self
        self.employees.append(new_employee)
      except ValueError as e:
        self.model.datacollector.no_worker_found += 1
        return

    # The least productive worker is fired if:
    # 1) the company employs workers
    # 2) the inventory at the end of the month is above the maximum stock
    elif self.stock > self.max_stock and len(self.employees) > 0:
      sorted(self.employees, key=lambda x: x.productivity)  
      self.model.datacollector.worker_fired += 1

      # remove the fired worker from the employee list and make her available for work again
      worker_fired = self.employees.pop()
      worker_fired.employer = None

  def distribute_profit(self):
    """
    Type:        Method 
    Description: Pay wages and distribute the remaining profits
    Execute:     Monthly 
    """
    # Pay wages based on the employees productivity 
    for employee in self.employees:
      # productivity is observed imperfectly. Wages are thus fluctuating 10% above and below actual productivity
      prod = employee.productivity
      wage = prod + np.random.uniform(prod - (prod /10), prod + (prod /10))
      if wage < 0:
        print('WTF')
      self.money -= wage
      employee.money += wage
      employee.income = wage 

    # Pay owner based on profits
    self.profit = self.money
    self.money  = 0

    # 
    # note: Puls not minus!!!!
    if self.profit >= 0:
      self.assets += 0.3 * self.profit
      self.owner.income = 0.7 * self.profit
      self.owner.money += 0.7 * self.profit
    
    elif self.profit < 0 and self.assets + self.profit >= 0:
      self.assets += self.profit
      self.owner.income = 0.7 * self.assets
      self.assets -= 0.7 * self.assets
      self.owner.money += 0.7 * self.assets
    
    else:
      self.assets += self.profit
      self.owner.income = 0
      self.owner.money += 0

      
  def step(self):
    """
    Type:        Method
    Description: Firm step function 
    Execute:     Monthly
    """
    # If it is the end of a month the month
    if self.model.schedule.steps%7 == 0:
      ## END OF MONTH
      # pay out wages and owner
      self.distribute_profit()

      ## BEGINNING OF MONTH
      # set the min_stock to the sales of previous month
      self.min_stock = self.sales * self.phi_l
      self.max_stock = self.sales * self.phi_u
      # reset sales and output for this month to zero
      self.sales = 0
      self.output = 0
      # set output level for this month
      self.set_labor()
      self.produce()
      # set price for this month
      self.set_price()

class Agent(mesa.Agent):
  """
  Type:         Mesa Agent Class
  Description:  Represents a household in the economy
  """
  def __init__(self, unique_id, model, village, income, firm, employer):
    super().__init__(unique_id, model)
    # parameters to be calibrated
    self.alpha = 0.9 # propensity to consume
    self.mu = 3
    self.sigma = 0.7

    # initialize geo-related characteristics
    self.village = village
    self.county = self.village.county
    self.pos = (self.village.pos[0] + np.random.uniform(-0.0003, 0.0003), # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
                self.village.pos[1] + np.random.uniform(-0.0003, 0.0003)) # it is randomly clustered around the village the agent belongs to
    
    # initialize other hh characteristics
    self.income = float(np.random.lognormal(self.mu, self.sigma, size=1) + 0.1)
    self.firm = firm 
    self.employer = employer
    self.best_dealers = []
    self.productivity = float(np.random.lognormal(self.mu, self.sigma, size=1) + 0.1)
    # initialize consumption related characteristics
    # @Extension: four categories:Food,Livestock, Non-Food Non-Durables, Durables, Temptation Goods
    # @Extension: distinguish between perishable and non-perishable good
    self.market_day = np.random.randint(0, 7) # day the agent goes to market. Note: bounds are included
    self.best_dealer_price = 1000 # agent remembers price of best dealer last week
    self.money = 1000 # for initial value estimate / retrive from data 
    # If not employed never gets updated!!!!!!!!!!!!!!!
    self.demand = 25 # @basic needs @TODO make dependent on hh size

  def find_dealer(self):
    """
    @Calibrate: list length of best_delaer
    Type:         Method 
    Description:  Maintains the list of best dealers  
    Used in:      buy_goods
    """
    # retrieve the list of all firms operating on the market
    potential_dealers = self.village.market.vendors

    # market needs at least three dealer on a given day to work
    if len(potential_dealers) < 5:
      return False
    
    # if the list of best dealers is empty try 3 'random dealers'
    if len(self.best_dealers) == 0:
      self.best_dealers = list(np.random.choice(potential_dealers, size=4, replace=False))

    # if price of best dealer increased compared to last week try a new dealer at random @spite
    # only checked from the second week onwards when the self.best_dealer_price contains a value
    elif self.best_dealers[0].price > self.best_dealer_price:

      # randomly select a new dealer from the dealers not already in the list and append it at the end 
      # note: np.random.choice returns a np. array; thus the index
      new_dealer = np.random.choice(list(set(potential_dealers) - set(self.best_dealers)), size=1, replace=False)[0]
      if self.best_dealers[-1].price > new_dealer.price:
        self.best_dealers[-1] =  new_dealer


    # sort the list of dealers according to the price they offer
    self.best_dealers = sorted(self.best_dealers, key=lambda x: x.price)
    self.best_dealer_price = self.best_dealers[0].price

    # return the first dealer in the list which has enough of the good on stock
    for dealer in self.best_dealers:
      if dealer.stock >= self.demand:
        return dealer

    self.model.datacollector.no_dealer_found += 1

    return False

  def trade(self, dealer):
    """
    @TODO implement partially satisfied demand see Lengnickp 109
    Type:        Method
    Description: Exchange of goods
    """
    # calculate the trade volume
    total_price =  dealer.price * self.demand
    
    # change the affected demand side variables
    self.money -= total_price

    # change the affected supply side variables
    dealer.stock -= self.demand
    dealer.sales += self.demand
    dealer.money += total_price 
    #@DELETE
    dealer.costumers.append(self)
    
    # save the transaction details in the model data collector
    self.model.datacollector.td_data.append({"step": self.model.schedule.steps, "parties": (self.unique_id, dealer.owner.unique_id), 
                                             "price": dealer.price, "amount": self.demand, "volume": total_price, 'market': self.village.market})

  def step(self):
    # hh step only needs to be executed on market day
    if  (self.model.schedule.steps - self.market_day)%7 == 0:
      dealer = self.find_dealer()
      # If there is an available dealer trade
      if dealer:
        self.trade(dealer)
        
    self.demand = pow(self.income, self.alpha) if self.income >= 0 else 0

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
     pass
  
  def __repr__(self):
     return f'Market: {self.unique_id} with costumers id: {[vendor.unique_id for vendor in self.vendors]}'


class Sugarscepe(mesa.Model):
  """
  Type:         Mesa Model Class
  Description:  Main Class for the simulation
  """
  def __init__(self, min_lat=-0.05 , max_lat=0.25, min_lon=34.00, max_lon=34.5, N=53000):
    print('init was called')

    # confine the geographic space of the grid to the study area
    self.x_min = min_lat
    self.x_max = max_lat
    self.y_min = min_lon
    self.y_max = max_lon
    self.grid = mesa.space.ContinuousSpace(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, torus=False)

    # initialize exogeneous population attributes
    self.N = N

    # initialize costum datacollector
    self.datacollector = Datacollector(self)

    # initialize scheduler
    self.schedule = mesa.time.RandomActivationByType(self)

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
    fm_dct = {row.hhid_key: Firm(unique_id=f"f_{row.hhid_key}", model=self, market=vl_dct[row.village_code].market,
                                 village=vl_dct[row.village_code])
              for row in self.df_fm.itertuples()}

    # create a list of agents based on the loaded df_hh
    hh_lst = [Agent(unique_id=row.hhid_key, model=self, village=vl_dct[row.village_code], income=row.p2_consumption,
                    firm=fm_dct.get(row.hhid_key, None), employer=fm_dct.get(row.hhid_key, None))
              for row in self.df_hh.itertuples()]

    # create N additional, syntetic hhs based on randomly chosen, existing hhs
    # Note: all syntetic hh have positive income: asserts that only firm owners can have non-positive income
    templates = np.random.choice([hh for hh in hh_lst], size=self.N, replace=True)
    for i, hh in enumerate(templates):
      new_instance = copy.copy(hh)
      new_instance.unique_id = f"HHS{i}_{hh.unique_id}"
      if np.random.random() < 0.4:
        firm = Firm(unique_id=f"HHf_{new_instance.unique_id}", model=self, market=new_instance.village.market,
                                 village=new_instance.village)
        new_instance.firm = firm
        new_instance.employer = firm

      else:
        new_instance.firm = None
        new_instance.employer = None

      hh_lst.append(new_instance)
      hh.pos = (hh.village.pos[0] + np.random.uniform(-0.0003, 0.0003), # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
                hh.village.pos[1] + np.random.uniform(-0.0003, 0.0003)) 

    # agentize the grid with hhs
    for hh in hh_lst:
      # place hh on grid and add it to schedule
      self.grid.place_agent(hh, hh.pos)
      self.schedule.add(hh)
      # add firms to schedule if applicable
      # note: done here so only firms with owners in the GE_HHLevel_ECMA.dta dataset are added
      if hh.firm != None:
        # add owner to firm
        hh.firm.owner = hh
        # add firm to schedule
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

    # create an attribute for quick acess to all markets in the model 
    self.all_markets = self.schedule.agents_by_type[Market].values()  

    # agentize the grid with vls
    for vl in vl_dct.values():
       # add vl to schedule
       self.schedule.add(vl)
       # initialize population 
       vl.population = [hh for hh in self.all_agents if hh.village == vl]
    
    # create an attribute for quick acess to all villages in the model 
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

    # collect data
    self.datacollector.collect_data()

    # for data collector to track the number of steps
    self.schedule.steps += 1 

    ## Conduct the intervention at step 200
    #if self.schedule.steps%200 == 0:
    #  self.intervention(1000)


  def intervention(self, amount):
    """
    Type:        Method
    Description: Simulates the unconditional cash transfer
    """
    for agent in self.all_agents:
      agent.income += amount


  def run_simulation(self, steps):
    """
    Type:        Method
    Description: Runs the simulation for the specified number of steps
    """
    start = timeit.default_timer()

    for i in range(steps):
      print(f"\rSimulating step: {i + 1} ({round((i + 1)*100/steps, 0)}% complete)", end="", flush=True)
      self.step()
    print("\n")

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

  def __repr__(self):
    return f'N Households: {len(self.all_agents)} \nN Firms: {len(self.all_firms)} \nN Villages: {len(self.all_villages)}\nN Markets: {len(self.all_markets)}'


def run_simulation(steps = 400):
  start = timeit.default_timer()

  model = Sugarscepe()
  model.run_simulation(steps)
  print(model)
  hh_data, fm_data, md_data, _ = model.datacollector.get_data()
  print(md_data[['average_stock', 'unemployment_rate', 'average_income', 'average_price', 
                'trade_volume', 'no_worker_found', 'no_dealer_found', 'worker_fired', ]].head(steps))
  
  
  stop = timeit.default_timer()
  print('Time: ', stop - start)  
  return model

if __name__ == "__main__":
    #cProfile.run("run_simulation()", filename="../data/profile_output.txt", sort='cumulative')
    
    # Create a pstats.Stats object from the profile file
    #profile_stats = pstats.Stats("../data/profile_output.txt")

    # Sort and print the top N entries with the highest cumulative time
    #profile_stats.strip_dirs().sort_stats('cumulative').print_stats(20)
    run_simulation()
    print('')