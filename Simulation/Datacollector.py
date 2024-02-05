import pandas as pd
import random
import numpy as np

def handle_none(agent):
  if agent != None:
    return agent.unique_id
  return None

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

    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income,[dealer.unique_id for dealer in agent.best_dealers],
                   agent.money, agent.demand, handle_none(agent.employer), handle_none(agent.firm))
                  for agent in self.model.all_agents]
    
    firm_data = [(step, firm.unique_id, firm.stock, firm.price, firm.output, firm.sales, firm.price * firm.sales,
                  firm.profit, firm.assets, len(firm.employees))
                 for firm in self.model.all_firms]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_hh_new = pd.DataFrame(agent_data, columns=['step','unique_id', 'village_id', 'lat', 'lon', "income", "best_dealers", "money", 
                                                  "demand", 'employer', 'owns_firm'])
    df_fm_new = pd.DataFrame(firm_data, columns=['step', 'unique_id', 'stock', 'price', 'output', 'sales', 'revenue', 'profit',
                                                 'assets', 'employees'])
 
     # Create a Pandas Dataframe from model data and reset their values
    df_md_new = {'step': step, 'no_worker_found': self.no_worker_found, 'no_dealer_found': self.no_dealer_found, 'worker_fired': self.worker_fired}
    
    # reset flow variables
    self.no_dealer_found = 0
    self.no_worker_found = 0
    self.worker_fired = 0

    # add dataframe of current step to the list
    self.hh_data.append(df_hh_new)
    self.fm_data.append(df_fm_new)
    self.md_data.append(df_md_new)

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
                df_td[df_td['step']==step]['volume'].sum(),
                df_td[df_td['step']==step]['price'].mean(), 
                df_td[df_td['step']==step]['amount'].sum(), 
                (sum(1 for item in df_hh[df_hh['step'] == step]['employer']  if item == None) / len(self.model.all_agents)),

                df_fm[df_fm['step']==step]['profit'].mean(),
                df_fm[df_fm['step']==step]['assets'].mean(),
                df_fm[df_fm['step']==step]['revenue'].mean(),
                df_fm[df_fm['step']==step]['stock'].mean(), 

                df_hh[df_hh['step']==step]['expenditure'].mean(),
                df_hh[df_hh['step']==step]['income'].mean(),
                df_hh[df_hh['step']==step]['income'].mean(),

                
                )
                for step in range(self.model.schedule.steps + 1)]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_md2 = pd.DataFrame(sm_data, columns=['step','Output', 'Employees', 'Trade_Volume', 'Price', 'Sales', 'Unemployment',                                   
                                            'Profit', 'Assets', 'Revenue','Stock', 
                                            'Expenditure', 'Money', 'Income', 
                                            ])
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
    self.td_data = []
    self.md_data = []

    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0


  def collect_data(self):
    """
    Type:         Datacollector Method
    Description:  Stores data generated in a given step as a pandas df
    """
    self.td_data = []

    return

  def get_calibration_data(self):

    data = [hh.demand for hh in self.model.all_agents if hh.village.market.saturation == 0 and hh.treated == 0]
    data = random.sample(data, k=2713)    
    return data

#Add all functionalities from collector 1 + hh fine grained data
class Validation_collector():
  def __init__(self, model):
    self.model = model
    self.td_data = []
    self.data = []

    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0
    self.count = 0


  def collect_data(self):
    """
    Type:         Datacollector Method
    Description:  Stores data generated in a given step as a pandas df
    """
    self.td_data = []
    
    #Just collect weekly data
    if self.model.schedule.steps % 7 != 0:
      return

### HH data
    hh_data = [(agent.income, agent.money, agent.demand, agent.firm, agent.employer, agent.treated, agent.village.market.saturation, agent.village.market.unique_id)
                for agent in self.model.all_agents]

### FM data
    fm_data = [(firm.assets, firm.profit, firm.revenue, firm.stock, firm.village.treated, firm.market.saturation, firm.market.unique_id) 
                 for firm in self.model.all_firms]
    
### MD data
    md_data = [(self.no_worker_found, self.no_dealer_found, self.worker_fired)]
    
    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0

    # Create DataFrames for agents and firms
    self.hh_df = pd.DataFrame(hh_data, columns=['Income', 'Money', 'Expenditure', 'Firm', 'Employer', 'Treated', 'Saturation', 'Market'])
    self.fm_df = pd.DataFrame(fm_data, columns=['Assets', 'Profit', 'Revenue', 'Stock','Treated', 'Saturation', 'Market'])

    hh_df_recipient = self.hh_df[self.hh_df['Treated'] == 1]
    hh_df_non_recipient = self.hh_df[self.hh_df['Treated'] == 0]

    fm_df_recipient = self.fm_df[self.fm_df['Treated'] == 1]
    fm_df_non_recipient = self.fm_df[self.fm_df['Treated'] == 0]


    df = {'step': self.count, 
          'Expenditure': self.hh_df['Expenditure'].mean(),      
          'Expenditure_Recipient': hh_df_recipient['Expenditure'].mean(),
          'Expenditure_Nonrecipient': hh_df_non_recipient['Expenditure'].mean(),

          'Money': self.hh_df['Money'].mean(),
          'Money_Recipient': hh_df_recipient['Money'].mean(),
          'Money_Nonrecipient': hh_df_non_recipient['Money'].mean(),

          'Income': self.hh_df['Income'].mean(),
          'Income_Recipient': hh_df_recipient['Income'].mean(),
          'Income_Nonrecipient': hh_df_non_recipient['Income'].mean(),

          'Profit': self.fm_df['Profit'].mean(),      
          'Profit_Recipient': fm_df_recipient['Profit'].mean(),
          'Profit_Nonrecipient': fm_df_non_recipient['Profit'].mean(),

          'Revenue': self.fm_df['Revenue'].mean(),
          'Revenue_Recipient': fm_df_recipient['Revenue'].mean(),
          'Revenue_Nonrecipient': fm_df_non_recipient['Revenue'].mean(),

          'Assets': self.fm_df['Assets'].mean(),
          'Assets_Recipient': fm_df_recipient['Assets'].mean(),
          'Assets_Nonrecipient': fm_df_non_recipient['Assets'].mean(),

          'Stock': self.fm_df['Stock'].mean(),
          'Stock_Recipient': fm_df_recipient['Stock'].mean(),
          'Stock_Nonrecipient': fm_df_non_recipient['Stock'].mean(),

          'Unemployment': len(self.hh_df[self.hh_df['Employer'].isna()]) / self.hh_df.shape[0],
          }

    self.data.append(df)
    self.count += 1
    return 
  
  def get_data(self):
    return  pd.DataFrame(self.data), self.hh_df, self.fm_df







#Add all functionalities from collector 1 + hh fine grained data
class Validation_collector2():
  def __init__(self, model):
    self.model = model
    self.td_data = []
    self.data = []
    self.map_data = []

    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0
    self.count = 0


  def collect_data(self):
    """
    Type:         Datacollector Method
    Description:  Stores data generated in a given step as a pandas df
    """
    self.td_data = []
    
    #Just collect weekly data
    if self.model.schedule.steps % 10 != 0:
      return
    
### HH data
    hh_data = [(self.count, agent.unique_id, agent.village.market.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.money, agent.demand, agent.firm, agent.employer, agent.treated, agent.village.market.saturation,)
                for agent in self.model.all_agents]

### FM data
    fm_data = [(firm.assets, firm.profit, firm.revenue, firm.stock, firm.village.treated, firm.market.saturation, firm.market.unique_id) 
                 for firm in self.model.all_firms]
    
### MD data
    md_data = [(self.no_worker_found, self.no_dealer_found, self.worker_fired)]
    
    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0

    # Create DataFrames for agents and firms
    self.hh_df = pd.DataFrame(hh_data, columns=['step', 'unique_id', 'Market', 'Village', 'lat', 'lon', 'Income', 'Money', 'Expenditure', 'Firm', 'Employer', 'Treated', 'Saturation',])
    self.fm_df = pd.DataFrame(fm_data, columns=['Assets', 'Profit', 'Revenue', 'Stock','Treated', 'Saturation', 'Market'])

    self.map_data.append(self.hh_df)

    hh_df_recipient = self.hh_df[self.hh_df['Treated'] == 1]
    hh_df_non_recipient = self.hh_df[self.hh_df['Treated'] == 0]

    fm_df_recipient = self.fm_df[self.fm_df['Treated'] == 1]
    fm_df_non_recipient = self.fm_df[self.fm_df['Treated'] == 0]


    df = {'step': self.count, 
          'Expenditure': self.hh_df['Expenditure'].mean(),      
          'Expenditure_Recipient': hh_df_recipient['Expenditure'].mean(),
          'Expenditure_Nonrecipient': hh_df_non_recipient['Expenditure'].mean(),
          'Expenditure_Lower': np.min(self.hh_df['Expenditure'].to_numpy()),#np.percentile(self.hh_df['Expenditure'].to_numpy(), 0),
          'Expenditure_Upper': np.max(self.hh_df['Expenditure'].to_numpy()),#np.percentile(self.hh_df['Expenditure'].to_numpy(), 100),
          
          'Money': self.hh_df['Money'].mean(),
          'Money_Recipient': hh_df_recipient['Money'].mean(),
          'Money_Nonrecipient': hh_df_non_recipient['Money'].mean(),
          'Money_Lower': np.percentile(self.hh_df['Money'].to_numpy(), 5),
          'Money_Upper': np.percentile(self.hh_df['Money'].to_numpy(), 95),

          'Income': self.hh_df['Income'].mean(),
          'Income_Recipient': hh_df_recipient['Income'].mean(),
          'Income_Nonrecipient': hh_df_non_recipient['Income'].mean(),
          'Income_Lower': np.percentile(self.hh_df['Income'].to_numpy(), 5),
          'Income_Upper': np.percentile(self.hh_df['Income'].to_numpy(), 95),

          'Profit': self.fm_df['Profit'].mean(),      
          'Profit_Recipient': fm_df_recipient['Profit'].mean(),
          'Profit_Nonrecipient': fm_df_non_recipient['Profit'].mean(),
          'Profit_Lower': np.percentile(self.fm_df['Profit'].to_numpy(), 5),
          'Profit_Upper': np.percentile(self.fm_df['Profit'].to_numpy(), 95),

          'Revenue': self.fm_df['Revenue'].mean(),
          'Revenue_Recipient': fm_df_recipient['Revenue'].mean(),
          'Revenue_Nonrecipient': fm_df_non_recipient['Revenue'].mean(),
          'Revenue_Lower': np.percentile(self.fm_df['Revenue'].to_numpy(), 5),
          'Revenue_Upper': np.percentile(self.fm_df['Revenue'].to_numpy(), 95),

          'Assets': self.fm_df['Assets'].mean(),
          'Assets_Recipient': fm_df_recipient['Assets'].mean(),
          'Assets_Nonrecipient': fm_df_non_recipient['Assets'].mean(),
          'Assets_Lower': np.percentile(self.fm_df['Assets'].to_numpy(), 5),
          'Assets_Upper': np.percentile(self.fm_df['Assets'].to_numpy(), 95),

          'Stock': self.fm_df['Stock'].mean(),
          'Stock_Recipient': fm_df_recipient['Stock'].mean(),
          'Stock_Nonrecipient': fm_df_non_recipient['Stock'].mean(),
          'Stock_Lower': np.percentile(self.fm_df['Stock'].to_numpy(), 5),
          'Stock_Upper': np.percentile(self.fm_df['Stock'].to_numpy(), 95),

          'Unemployment': len(self.hh_df[self.hh_df['Employer'].isna()]) / self.hh_df.shape[0],
          }

    self.data.append(df)
    self.count += 1
    return 
  
  def get_data(self):
    return   pd.concat(self.map_data, axis=0), pd.DataFrame(self.data)


