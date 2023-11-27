import mesa
import numpy as np
from read_data import create_agent_data
from data_collector import Datacollector
import timeit
import copy

# @TODO 
# change initial variables to none not zero

class Firm(mesa.Agent):
  """
  Type:        Mesa Agent Class
  Description: Implements firm entity
  """
  def __init__(self, unique_id, model, market, village):
    super().__init__(unique_id, model)

    # firm properties
    self.id = unique_id
    self.owner = None
    self.market = market
    self.village = village
    self.productivity = 1.03
    # price
    self.price = np.random.uniform(1, 10) # price in current month (initialized randomly)
    self.marginal_cost = 1# labor is payed its productivity 
    self.max_price = 11 
    self.theta = 0.8 # probability for price change
    self.nu = 0.3 # rate for price change @Calibrate
    self.phi_l = 0.1 # phi_l * sales = minimal stock @Calibrate
    self.phi_u = 1 # phi_u * sales = max stock for @Calibrate
    self.employees = [] 
    # inventory 
    self.min_stock = 0 # 10% percent of last months sales 
    self.max_stock = 0 # 150% percent of last months sales 
    self.stock = 300
    self.output = 0 # quantity produced this month
    self.sales = 0  # quantity sold this month
    # profit
    self.money = 0
    self.assets = 200
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
        current_price *= (1 + self.nu) #self.nu * (current_price - self.marginal_cost)

    # price is decresed with probability theta if the follwoing conditions are met:
    # 1) the stock exceeds the maximum inventory
    # 2) price last month was strictly larger than the marginal cost
    elif self.stock > self.max_stock: #and current_price * (1-self.nu) > self.marginal_cost:
      if np.random.uniform(0, 1) <= self.theta:
        current_price -= self.nu * (current_price - self.marginal_cost)

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
    self.output = amount
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
      worker_fired.income = 0

  def distribute_profit(self):
    """
    Type:        Method 
    Description: Pay wages and distribute the remaining profits
    Execute:     Monthly 
    """
    # Pay wages based on the employees productivity 
    for employee in self.employees:
      #@Implement
      # productivity is observed imperfectly. Wages are thus fluctuating 10% above and below actual productivity
      wage = employee.productivity

      self.money -= wage
      employee.money += wage
      employee.income = wage 

    # Pay owner based on profits
    self.profit = self.money
    self.money  = 0

    # If profit positiv: pay part of it to owner
    # note: Plus not minus
    # @Make this a parameter
    if self.profit >= 0:
      reserves = 0.1 if self.assets >= 0 else 0.9
      self.assets += reserves * self.profit
      self.owner.income = (1 - reserves) * self.profit
      self.owner.money += (1 - reserves) * self.profit

    
    elif self.profit < 0 and self.assets + self.profit >= 0:
      self.assets += self.profit

      payout = 0.1 * self.assets
      self.owner.income = payout
      self.owner.money += payout
      self.assets -= payout
    
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
  def __init__(self, unique_id, model, village, firm, employer):
    super().__init__(unique_id, model)
    # parameters to be calibrated
    self.alpha = 0.8 # propensity to consume
    self.mu = 3
    self.sigma = 0.7

    # initialize geo-related characteristics
    self.village = village
    self.county = self.village.county
    self.pos = (self.village.pos[0] + np.random.uniform(-0.0003, 0.0003), # pos is a tuple of shape (lat, lon) as used for continuous_grid in mesa
                self.village.pos[1] + np.random.uniform(-0.0003, 0.0003)) # it is randomly clustered around the village the agent belongs to
    
    # initialize other hh characteristics
    self.income = 0 
    self.firm = firm 
    self.employer = employer
    self.best_dealers = []
    self.productivity = float(np.random.lognormal(self.mu, self.sigma, size=1) + 1)
    self.treated = 0

    # initialize consumption related characteristics
    self.market_day = np.random.randint(0, 7) # day the agent goes to market. Note: bounds are included
    self.best_dealer_price = 10 # agent remembers price of best dealer last week
    self.money = 100 # household liquidity
    self.demand = 0 

  def find_dealer(self):
    """
    @Calibrate: list length of best_delaer
    Type:         Method 
    Description:  Maintains the list of best dealers  
    Used in:      buy_goods
    """
    # retrieve the list of all firms operating on the market
    potential_dealers = self.village.market.vendors
    np.random.shuffle(potential_dealers)

    # market needs at least 5 dealers on a given day to work
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

    # note: the if statment is necessary because of float rounding errors.
    # 
    self.demand = pow(self.money, self.alpha) if self.money > 0 else 0
    
    # return the list of best dealers extended by all potential dealers in case demand is not satisfied
    return self.best_dealers + potential_dealers


  def trade(self, dealer, amount):
    """
    Type:        Method
    Description: Exchange of goods
    """
    # calculate trade volume
    total_price =  dealer.price * amount

    # change the affected demand side variables
    self.money -= total_price

    # change the affected supply side variables
    dealer.stock -= amount
    dealer.sales += amount
    dealer.money += total_price 
    #@DELETE
    dealer.costumers.append(self)
    
    # save the transaction details in the model data collector
    self.model.datacollector.td_data.append({"step": self.model.schedule.steps, "parties": (self.unique_id, dealer.owner.unique_id), 
                                             "price": dealer.price, "amount": amount, "volume": total_price, 'market': self.village.market.unique_id,
                                              'from': self.village.unique_id, 'to': dealer.owner.village.unique_id})

  def step(self):
    # hh step only needs to be executed on market day
    if  (self.model.schedule.steps - self.market_day)%7 == 0:
      # get the list of best dealers
      best_dealers = self.find_dealer()
      demand = self.demand

      # iterate through the list of best dealers until demand is satisfied
      for dealer in best_dealers:
        
        # account for the case where the agent cannot afford her demand
        amount = max(0, min(self.money / dealer.price, demand))
        demand = amount

        # the remaining demand is smaller than 0.25. The agent does not contiune buying
        if amount <= 0.25:
          return

        # if dealer doesn't have enough on stock: buy all that remains
        if dealer.stock - demand < 0:
          amount = min(dealer.stock, amount)

        # the dealer has nothing left on stock. The agents goes to the next dealer
        if amount <= 0:
          continue

        self.trade(dealer, amount)
        demand -= amount

      
      self.model.datacollector.no_dealer_found +=1 
        

  def __repr__(self):
    return f'Agent: {self.unique_id} at coordinates: {self.pos} in county {self.county}'


class Village(mesa.Agent):
    """
    Type:         Mesa Agent Class
    Description:  Physical location of households. Deterimines social environment of agents
                  level at which spillovers occur
    """
    def __init__(self, unique_id, model, pos, county, market, treated, saturation):
      super().__init__(unique_id, model)
      self.pos = pos
      self.county = county
      self.market = market
      self.population = None
      self.treated = 0
      self.saturation = saturation

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
    self.villages = None
    self.saturation = 0
  
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
    vl_dct = {row.village_code : Village(unique_id=int(row.village_code), model=self, pos=row.pos, county=row.county, 
                                         market=mk_dct[row.market_id], treated=row.treat, saturation= row.hi_sat)
              for row in self.df_vl.itertuples()}
    
    # create a dictionary of firms based on loaded df_fm 
    fm_dct = {row.hhid_key: Firm(unique_id=f"f_{row.hhid_key}", model=self, market=vl_dct[row.village_code].market,
                                 village=vl_dct[row.village_code])
              for row in self.df_fm.itertuples()}

    # create a list of agents based on the loaded df_hh
    hh_lst = [Agent(unique_id=row.hhid_key, model=self, village=vl_dct[row.village_code],
                    firm=fm_dct.get(row.hhid_key, None), employer=fm_dct.get(row.hhid_key, None))
              for row in self.df_hh.itertuples()]

    # create N additional, syntetic hhs based on randomly chosen, existing hhs
    # Note: all syntetic hh have positive income: asserts that only firm owners can have non-positive income
    templates = np.random.choice([hh for hh in hh_lst], size=self.N, replace=True)
    for i, hh in enumerate(templates):
      new_instance = copy.copy(hh)
      new_instance.unique_id = f"HHS{i}_{hh.unique_id}"
      if np.random.random() < 0.2:
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
    self.all_agents = list(self.schedule.agents_by_type[Agent].values())
  
    # create an attribute for quick acess to all firms in the model 
    self.all_firms = list(self.schedule.agents_by_type[Firm].values())

    # agentize the grid with mks
    for mk in mk_dct.values():
       # add mk to schedule
       self.schedule.add(mk)
       # store vendors operating on the market
       mk.vendors = [firm for firm in self.all_firms if firm.market == mk] 
       # store villages vendors are coming from (for UCT)
       mk.villages = [vendor.village for vendor in mk.vendors]

    # create an attribute for quick acess to all markets in the model 
    self.all_markets = list(self.schedule.agents_by_type[Market].values())  

    # agentize the grid with vls
    for vl in vl_dct.values():
       # add vl to schedule
       self.schedule.add(vl)
       # initialize population 
       vl.population = [hh for hh in self.all_agents if hh.village == vl]
    
    # create an attribute for quick acess to all villages in the model 
    self.all_villages = list(self.schedule.agents_by_type[Village].values())  


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

    # Conduct the intervention at step 800
    if self.schedule.steps == 800:

      # assign treatment status 
      self.assign_treatement_status()

      # distribute the token (USD 150 PPP)
      self.intervention(50) 

    if self.schedule.steps == 920:
       # distribute first handoud (USD 860 PPP)
       self.intervention(300)
    
    if self.schedule.steps == 1100:
       # distribute first handoud (USD 860 PPP)
       self.intervention(300)
    

  def assign_treatement_status(self):
    """
    Type:        Method
    Description: Assigns saturation and treatment status on market, village and agnet level
    """

### Level 1 randomization

    # assign high saturation status to 30 random markets
    high_sat_mk = np.random.choice(self.all_markets, size=30, replace=False)
    for mk in high_sat_mk:
      setattr(mk, 'saturation', 1)

### Level 2 randomizaton

    # assign control status to 2/3 of villages in low saturation mks 
    # assign treatment status to 2/3 of villages in high saturation mks 
    for mk in self.all_markets:
      # choose treatment villages fraction depending on market saturation status
      treat_frac = 2/3 if mk.saturation == 1 else 1/3
      mk_villages = mk.villages
      treatment_villages = np.random.choice(mk_villages, int(len(mk_villages) * treat_frac), replace=False)

      # assign treatment status to the selected villages
      for vl in treatment_villages:
        setattr(vl, 'treated', 1)

### Level 3 randomization 

    # for each treatment village identify the 30 poorest households
    self.treated_agents = []
    for vl in treatment_villages:
      sorted_population = sorted(vl.population, key=lambda x: x.money)
      self.treated_agents.extend(sorted_population[:18])
    
    # assign treatment status to the selected agents
    for agent in self.treated_agents:
      setattr(agent, 'treated', 1)

    print(f"# treated hhs: {len(self.treated_agents)}")


  def intervention(self, amount):
    """
    Type:        Method
    Description: Simulates the unconditional cash transfer
    """
    for agent in self.treated_agents:
      agent.money += amount


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


def run_simulation(steps = 10):
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

