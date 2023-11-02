import pandas as pd

#@TODO make both collectors more sparse once known what data is
class Datacollector():
  """
  Type:        Helper Class 
  Description: Collects data at the agent and model level
  """
  def __init__(self, model):
    self.model = model
    self.hh_data = []
    self.fm_data = []
    self.md_data = []
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
    # collect hh and firm data for the current step
    step = self.model.schedule.steps

    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.best_dealers,
                   agent.consumption, agent.money, agent.demand, agent.employer, agent.firm)
                  for agent in self.model.all_agents]
    
    firm_data = [(step, firm.unique_id, firm.stock, firm.price, firm.money, firm.output, firm.sales, firm.price * firm.sales,
                  len(firm.employees), len(set(firm.costumers)))
                 for firm in self.model.all_firms]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_hh_new = pd.DataFrame(agent_data, columns=['step','unique_id', 'village_id', 'lat', 'lon', "income", "best_dealers", "consumption", "money", 
                                                  "demand", 'employer', 'owns_firm'])
    df_fm_new = pd.DataFrame(firm_data, columns=['step', 'unique_id', 'stock', 'price', 'money', 'output', 'sales', 'revenue', 'employees', 
                                                 'costumers'])
 
     # Create a Pandas Dataframe from model data and reset their values
    df_md_new = {'step': step, 'no_worker_found': self.no_worker_found, 'no_dealer_found': self.no_dealer_found, 'worker_fired': self.worker_fired}
    self.no_dealer_found = 0
    self.no_worker_found = 0
    self.worker_fired = 0

    # add dataframe of current step to the list
    self.hh_data.append(df_hh_new)
    self.fm_data.append(df_fm_new)
    self.md_data.append(df_md_new)

  # @TODO specify which frames to 
  def get_data(self):
    """
    Type:        Datacollector Method
    Description: Concatenates all collected data into DataFrames for hh, fm and md
    Exectuted:   Once
    """
    # Concatenate the collected dataframes
    df_hh = pd.concat(self.hh_data, axis=0)
    df_fm = pd.concat(self.fm_data, axis=0)
    df_td = pd.DataFrame(self.td_data)
    df_md1 = pd.DataFrame(self.md_data)

    # Get summary statistics from the concatanated dataframes
    sm_data = [(step,  
                df_fm[df_fm['step']==step]['output'].mean(),
                df_fm[df_fm['step']==step]['employees'].mean(), 
                df_fm[df_fm['step']==step]['stock'].mean(), 
                df_td[df_td['step']==step]['amount'].sum(), 
                (sum(1 for item in df_hh[df_hh['step'] == step]['employer']  if item == None) / len(self.model.all_agents)),
                df_hh.loc[(df_hh.step==step)]['income'].mean(), #& (pd.isna(df_hh.owns_firm)
                df_td[df_td['step']==step]['price'].mean(), 
                df_td[df_td['step']==step]['volume'].sum(),
                # @TODO make this directly in collector to find fiv by zero
                #df_td[df_td['step']==step]['amount'].sum() / 
                (df_hh[df_hh['step']==step]['demand'].sum()/7),
                df_fm[df_fm['step']==step]['output'].sum())
                for step in range(self.model.schedule.steps + 1)]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_md2 = pd.DataFrame(sm_data, columns=['step','total_output', 'average_employees', 'average_stock', 'total_sales', 'unemployment_rate',
                                            'average_income', 'average_price', 'trade_volume', 'demand_satisfied', 'output'])
    
    # Put all model level data into one dataframe
    df_md = pd.merge(df_md1, df_md2, on='step')

    return df_hh, df_fm, df_md, df_td
  




class Sparse_collector():
  """
  Type:        Helper Class 
  Description: Sparse version of datacollector
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
    Description:  Stores data generated in a given step as a pandas df
    """
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
    df_md = {'step': step, 'no_worker_found': self.no_worker_found, 'no_dealer_found': self.no_dealer_found, 'worker_fired': self.worker_fired}
    self.no_dealer_found = 0
    self.no_worker_found = 0
    self.worker_fired = 0

    # Get summary statistics from the dataframes
    sm_data = [(step,  
                df_fm[df_fm['step']==step]['stock'].mean(), 
                (sum(1 for item in df_hh[df_hh['step'] == step]['employer']  if item == None) / len(self.model.all_agents)),
                df_hh.loc[(df_hh.step==step)]['income'].mean())]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_sm = pd.DataFrame(sm_data, columns=['step','average_stock','unemployment_rate','average_income'])
    self.data.append(df_sm)
  
      
  def get_data(self):
    df_sm = pd.concat(self.data, axis=0)
    return df_sm