#%%%
import numpy as np
import pandas as pd
import ABM
import timeit
import dill 
import cProfile # move to Test.py
import pstats   # move to Test.py
['average_stock', 'unemployment_rate', 'average_income']

class sparse_collector():
  """
  Type:        Helper Class 
  Description: Collects data at the agent and model level
  """
  def __init__(self, model):
    self.model = model
    self.data = []
    self.td_data = []

    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0


  def collect_data(self):
    """
    Type:         Datacollector Method
    Description:  Stores all agent-level data generated in a step as a pandas df
    Executed:     Daily
    """
    self.hh_data = []
    self.td_data = []

    # collect hh and firm data for the current step
    step = self.model.schedule.steps

    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.best_dealers,
                   agent.consumption, agent.money, agent.demand, agent.employer, agent.firm)
                  for agent in self.model.all_agents]
    
    firm_data = [(step, firm.unique_id, firm.stock, firm.price, firm.money, firm.output, firm.sales, firm.price * firm.sales,
                  len(firm.employees), len(set(firm.costumers)))
                 for firm in self.model.all_firms]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_hh = pd.DataFrame(agent_data, columns=['step','unique_id', 'village_id', 'lat', 'lon', "income", "best_dealers", "consumption", "money", 
                                              'demand', 'employer', 'owns_firm'])
    
    df_fm = pd.DataFrame(firm_data, columns=['step', 'unique_id', 'stock', 'price', 'money', 'output', 'sales', 'revenue', 'employees', 
                                             'costumers'])
 
    # Create a Pandas Dataframe from model data and reset their values
    df_md_new = {'step': step, 'no_worker_found': self.no_worker_found, 'no_dealer_found': self.no_dealer_found, 'worker_fired': self.worker_fired}

    # Get summary statistics from the dataframes
    sm_data = [(step,  
                df_fm[df_fm['step']==step]['stock'].mean(), 
                (sum(1 for item in df_hh[df_hh['step'] == step]['employer']  if item == None) / len(self.model.all_agents)),
                df_hh.loc[(df_hh.step==step)]['income'].mean())]
    
    # Create a Pandas DataFrames from the list comprehensions
    sm_data = pd.DataFrame(sm_data, columns=['step','average_stock','unemployment_rate','average_income'])
    self.data.append(sm_data)

    
  def get_data(self):
    df_sm = pd.concat(self.data, axis=0)
    return df_sm

class Model(ABM.Sugarscepe):
  def __init__(self):
    super().__init__()
    self.datacollector = sparse_collector(self)

  def set_parameters(self, fm_sample):
    """
    Type:        Method
    Description: Overwites the model parameters as specified in the 'parameters' argument
    """
    for firm in self.all_firms:
      for parameter, value in fm_sample.items():
        setattr(firm, parameter, value)

  def run_simulation(self, steps):

    """
    Type: Method
    Description: overwrite run simulaiton in parent class to avoid print statement
    """
    for i in range(steps):
      self.step()

  
def run_model(steps):
  simulation = Model()
  with open('../data/simulation_initial_state.dill', 'wb') as file:
    dill.dump(simulation, file)

  for i in range(steps):
    start = timeit.default_timer()
    with open('../data/simulation_initial_state.dill', 'rb') as file:
      sim_copy = dill.load(file)

    sim_copy.run_simulation(100)
    df_sm = sim_copy.datacollector.get_data()
    print(df_sm)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  

run_model(50)
