import pandas as pd
import numpy as np


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
                   agent.money, agent.demand, agent.employer, agent.firm)
                  for agent in self.model.all_agents]
    
    firm_data = [(step, firm.unique_id, firm.stock, firm.price, firm.money, firm.output, firm.sales, firm.price * firm.sales,
                  firm.profit, firm.assets, len(firm.employees))
                 for firm in self.model.all_firms]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_hh_new = pd.DataFrame(agent_data, columns=['step','unique_id', 'village_id', 'lat', 'lon', "income", "best_dealers", "money", 
                                                  "demand", 'employer', 'owns_firm'])
    df_fm_new = pd.DataFrame(firm_data, columns=['step', 'unique_id', 'stock', 'price', 'money', 'output', 'sales', 'revenue', 'profit',
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
                df_fm[df_fm['step']==step]['profit'].mean(),
                df_td[df_td['step']==step]['amount'].sum(), 
                (sum(1 for item in df_hh[df_hh['step'] == step]['employer']  if item == None) / len(self.model.all_agents)),
                df_hh[df_hh['step']==step]['income'].mean(),
                df_td[df_td['step']==step]['price'].mean(), 
                df_td[df_td['step']==step]['volume'].sum(),
                # @TODO make this directly in collector to find fiv by zero
                df_td[df_td['step']==step]['amount'].sum() / (df_hh[df_hh['step']==step]['demand'].sum()/7),
                df_fm[df_fm['step']==step]['output'].sum())
                for step in range(self.model.schedule.steps + 1)]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_md2 = pd.DataFrame(sm_data, columns=['step','total_output', 'average_employees', 'average_stock', 'average_profit', 'total_sales', 
                                            'unemployment_rate', 'average_income', 'average_price', 'trade_volume', 
                                            'demand_satisfied', 'output'])
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
    step = self.model.schedule.steps

    self.md_data.append( {'step': step, 'no_worker_found': self.no_worker_found, 'no_dealer_found': self.no_dealer_found, 'worker_fired': self.worker_fired})
    
    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0
    return

  def get_calibration_data(self):
    # @TODO add
    # Total output 
    # Gini coefficient
    # Income if measured in data?
    # wage

    hh_data = [(hh.employer, hh.firm, hh.demand, hh.income, hh.money, hh.productivity) 
                for hh in self.model.all_agents]
    
    fm_data = [(firm.unique_id, firm.stock, firm.output, firm.profit, firm.sales, firm.assets, len(firm.employees), len(firm.costumers), firm.market.unique_id, firm.price)
                for firm in self.model.all_firms]
    
    df_hh = pd.DataFrame(hh_data, columns=['employer', 'firm', 'demand', 'income', 'money', 'productivity'])
    
    df_fm = pd.DataFrame(fm_data, columns=['id', 'stock', 'output', 'profit', 'sales', 'assets', 'employees', 
                                           'costumers', 'market_id', 'price'])

    sm_data = [((sum(1 for item in df_hh['employer']  if item == None) / len(self.model.all_agents)), # umemployment_rate
                df_fm['employees'].mean(),    # employees average
                df_fm['employees'].var(),     # employees variance
                df_fm['profit'].mean(),       # profit averag
                df_fm['profit'].var(),        # profit variance
                df_fm['sales'].mean(),        # revenue average
                df_fm['sales'].var(),         # revenue variance
                df_fm['stock'].mean(),        # stock average 
                df_fm['stock'].var(),         # stock variance
                df_hh['demand'].mean(),  # consumption averrage
                df_hh['demand'].var())]  # consumption variance
    

    df_md = pd.DataFrame(self.md_data)
    df_td = pd.DataFrame(self.td_data)

    return np.array(sm_data).flatten(), df_hh, df_fm, df_md, df_td





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
    if self.model.schedule.steps % 7 != 0:
      return

### HH data
    hh_data = [(agent.income, agent.money, agent.demand, agent.firm, agent.employer, agent.treated)
                for agent in self.model.all_agents]

### FM data
    fm_data = [(firm.assets, firm.profit, firm.sales * firm.price, firm.stock, firm.village.treated) 
                 for firm in self.model.all_firms]
    
### MD data
    md_data = [(self.no_worker_found, self.no_dealer_found, self.worker_fired)]
    
    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0

    # Create DataFrames for agents and firms
    hh_df = pd.DataFrame(hh_data, columns=['Income', 'Money', 'Demand', 'Firm', 'Employer', 'Treated'])
    fm_df = pd.DataFrame(fm_data, columns=['Assets', 'Profit', 'Revenue', 'Stock','Treated'])
    #md_df = pd.DataFrame(md_data, columns=['No_worker_found', 'no_dealer_found', 'worker_fired' ])

    df = {'step': self.count, 
          'Income_Treated': hh_df[hh_df['Treated'] == 1]['Income'].mean(),
          'Income_Control': hh_df[hh_df['Treated'] == 0]['Income'].mean(),
          'Money_Treated': hh_df[hh_df['Treated'] == 1]['Money'].mean(),
          'Money_Control': hh_df[hh_df['Treated'] == 0]['Money'].mean(),          
          'Demand_Treated': hh_df[hh_df['Treated'] == 1]['Demand'].mean(),
          'Demand_Control': hh_df[hh_df['Treated'] == 0]['Demand'].mean(),
          'Assets_Treated': fm_df[fm_df['Treated'] == 1]['Assets'].mean(),
          'Assets_Control': fm_df[fm_df['Treated'] == 0]['Assets'].mean(),
          'Revenue_Treated': fm_df[fm_df['Treated'] == 1]['Revenue'].mean(),
          'Revenue_Control': fm_df[fm_df['Treated'] == 0]['Revenue'].mean(),          
          'Profit_Treated': fm_df[fm_df['Treated'] == 1]['Profit'].mean(),
          'Profit_Control': fm_df[fm_df['Treated'] == 0]['Profit'].mean(),
          'Stock_Treated': fm_df[fm_df['Treated'] == 1]['Stock'].mean(),
          'Stock_Control': fm_df[fm_df['Treated'] == 0]['Stock'].mean(),
          'Unemployment': len(hh_df[hh_df['Employer'].isna()]) / hh_df.shape[0],
          }

    self.data.append(df)
    self.count += 1
    return 
  
  def get_data(self):
    return  pd.DataFrame(self.data)


