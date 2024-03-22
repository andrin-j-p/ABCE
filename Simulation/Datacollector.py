import pandas as pd
import random
import numpy as np


class Datacollector():
  """
  Type:        Helper Class
  Description: Collects agent and aggregate level data. Used in visualize.py
  """
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
    Type:         Method
    Description:  Stores data generated in a given step as a pandas df
    """
    # Trade data is not used and thus emptied to save memory
    self.td_data = []
    
    # Just collect weekly data for performance purposes
    if self.model.schedule.steps % 7 != 0:
      return
    
### HH data
    hh_data = [(self.count, agent.unique_id, agent.village.market.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.money, agent.demand, agent.firm, agent.employer, agent.treated, agent.village.market.saturation,)
                for agent in self.model.all_agents]

### FM data
    fm_data = [(firm.assets, firm.profit, firm.revenue, firm.stock, firm.village.treated, firm.market.saturation, firm.market.unique_id) 
                 for firm in self.model.all_firms]
    
### MD data
    # Collect labor market data
    md_data = [(self.no_worker_found, self.no_dealer_found, self.worker_fired)]
    
    # Reset labor market statistics to zero at the end of the week
    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0

    # Create DataFrames for agents and firms
    self.hh_df = pd.DataFrame(hh_data, columns=['step', 'unique_id', 'Market', 'Village', 'lat', 'lon', 'Income', 'Money', 'Expenditure', 'Firm', 'Employer', 'Treated', 'Saturation',])
    self.fm_df = pd.DataFrame(fm_data, columns=['Assets', 'Profit', 'Revenue', 'Stock','Treated', 'Saturation', 'Market'])

    # Agent level data used for the animation frame
    self.map_data.append(self.hh_df)

    # Split dataframe into one for recipient and one for non-recipient households 
    hh_df_recipient = self.hh_df[self.hh_df['Treated'] == 1]
    hh_df_non_recipient = self.hh_df[self.hh_df['Treated'] == 0]

    # Split dataframe into one for firms in treated and untreated villages
    fm_df_recipient = self.fm_df[self.fm_df['Treated'] == 1]
    fm_df_non_recipient = self.fm_df[self.fm_df['Treated'] == 0]

    # Calculate aggregate outcomes
    df = {'step': self.count, 
          'Expenditure': self.hh_df['Expenditure'].mean(),      
          'Expenditure Recipient': hh_df_recipient['Expenditure'].mean(),
          'Expenditure Nonrecipient': hh_df_non_recipient['Expenditure'].mean(),
          'Expenditure Lower': np.min(self.hh_df['Expenditure'].to_numpy()),#np.percentile(self.hh_df['Expenditure'].to_numpy(), 0),
          'Expenditure Upper': np.max(self.hh_df['Expenditure'].to_numpy()),#np.percentile(self.hh_df['Expenditure'].to_numpy(), 100),
          
          'Money': self.hh_df['Money'].mean(),
          'Money Recipient': hh_df_recipient['Money'].mean(),
          'Money Nonrecipient': hh_df_non_recipient['Money'].mean(),
          'Money Lower': np.percentile(self.hh_df['Money'].to_numpy(), 5),
          'Money Upper': np.percentile(self.hh_df['Money'].to_numpy(), 95),

          'Income': self.hh_df['Income'].mean(),
          'Income Recipient': hh_df_recipient['Income'].mean(),
          'Income Nonrecipient': hh_df_non_recipient['Income'].mean(),
          'Income Lower': np.percentile(self.hh_df['Income'].to_numpy(), 5),
          'Income Upper': np.percentile(self.hh_df['Income'].to_numpy(), 95),

          'Profit': self.fm_df['Profit'].mean(),      
          'Profit Recipient': fm_df_recipient['Profit'].mean(),
          'Profit Nonrecipient': fm_df_non_recipient['Profit'].mean(),
          'Profit Lower': np.percentile(self.fm_df['Profit'].to_numpy(), 5),
          'Profit Upper': np.percentile(self.fm_df['Profit'].to_numpy(), 95),

          'Revenue': self.fm_df['Revenue'].mean(),
          'Revenue Recipient': fm_df_recipient['Revenue'].mean(),
          'Revenue Nonrecipient': fm_df_non_recipient['Revenue'].mean(),
          'Revenue Lower': np.percentile(self.fm_df['Revenue'].to_numpy(), 5),
          'Revenue Upper': np.percentile(self.fm_df['Revenue'].to_numpy(), 95),

          'Assets': self.fm_df['Assets'].mean(),
          'Assets Recipient': fm_df_recipient['Assets'].mean(),
          'Assets Nonrecipient': fm_df_non_recipient['Assets'].mean(),
          'Assets Lower': np.percentile(self.fm_df['Assets'].to_numpy(), 5),
          'Assets Upper': np.percentile(self.fm_df['Assets'].to_numpy(), 95),

          }

    self.data.append(df)
    self.count += 1
    return 
  
  def get_data(self):
    """
    Type:        Method
    Description: Concatenates data collected after the simulation terminates
    """
    return  pd.concat(self.map_data, axis=0), pd.DataFrame(self.data)


class Validation_collector():
  """
  Type:        Class 
  Description: Less detailed version of Datacollector. Used in validation.py
  """
  def __init__(self, model):
    self.model = model
    self.td_data = []
    self.data = []
    self.no_worker_found = 0
    self.no_dealer_found = 0
    self.worker_fired = 0
    self.residual_demand = 0
    self.count = 0

  def collect_data(self):
    """
    Type:         Datacollector Method
    Description:  Stores data generated in a given step as a pandas df
    """
    td_data = pd.DataFrame(self.td_data)
    total_volume = td_data['volume'].sum()

    # Trade data is no longer used and thus emptied to save memory
    self.td_data = []
    
    # Just collect weekly data for performance purposes
    if self.model.schedule.steps % 7 != 0:
      return

### HH data
    hh_data = [(agent.income, agent.money, agent.demand, agent.firm, agent.employer, agent.demand, agent.treated, agent.eligible, agent.village.market.saturation, agent.village.treated, agent.village.market.unique_id)
                for agent in self.model.all_agents]

### FM data
    fm_data = [(firm.assets, firm.profit, firm.revenue, firm.stock, firm.wage_bill, firm.price, len(firm.employees), firm.village.treated, firm.market.saturation, firm.market.unique_id) 
                 for firm in self.model.all_firms]

    # Create DataFrames for agents and firms
    self.hh_df = pd.DataFrame(hh_data, columns=['Income', 'Assets', 'Expenditure', 'Firm', 'Employer', 'Demand', 'Treated', 'Eligible','Saturation', 'Village', 'Market'])
    self.fm_df = pd.DataFrame(fm_data, columns=['Money', 'Profit', 'Revenue', 'Stock', 'Wage', 'Price', 'Employees', 'Treated', 'Saturation', 'Market'])

    # Split dataframe into one for recipient and one for non-recipient households 
    hh_df_recipient = self.hh_df[self.hh_df['Treated'] == 1]
    hh_df_non_recipient = self.hh_df[self.hh_df['Treated'] == 0]

    # Split dataframe into one for firms in treated and untreated villages
    fm_df_recipient = self.fm_df[self.fm_df['Treated'] == 1]
    fm_df_non_recipient = self.fm_df[self.fm_df['Treated'] == 0]

    # Calculate aggregate outcomes
    df = {'step': self.count, 
          'Expenditure': self.hh_df['Expenditure'].mean(),      
          'Expenditure_Recipient': hh_df_recipient['Expenditure'].mean(),
          'Expenditure_Nonrecipient': hh_df_non_recipient['Expenditure'].mean(),

          'Assets': self.hh_df['Assets'].mean(),
          'Assets_Recipient': hh_df_recipient['Assets'].mean(),
          'Assets_Nonrecipient': hh_df_non_recipient['Assets'].mean(),

          'Income': self.hh_df['Income'].mean(),
          'Income_Recipient': hh_df_recipient['Income'].mean(),
          'Income_Nonrecipient': hh_df_non_recipient['Income'].mean(),

          'Profit': self.fm_df['Profit'].mean(),      
          'Profit_Recipient': fm_df_recipient['Profit'].mean(),
          'Profit_Nonrecipient': fm_df_non_recipient['Profit'].mean(),

          'Revenue': self.fm_df['Revenue'].mean(),
          'Revenue_Recipient': fm_df_recipient['Revenue'].mean(),
          'Revenue_Nonrecipient': fm_df_non_recipient['Revenue'].mean(),

          'Stock': self.fm_df['Stock'].mean(),
          'Stock_Recipient': fm_df_recipient['Stock'].mean(),
          'Stock_Nonrecipient': fm_df_non_recipient['Stock'].mean(),
          
          'Wage': self.fm_df['Wage'].mean(),
          'Wage_Recipient': fm_df_recipient['Wage'].mean(),
          'Wage_Nonrecipient': fm_df_non_recipient['Wage'].mean(),

          'Price': self.fm_df['Price'].mean(),
          'Trade Volume': total_volume, 
          'Average Number of Employees': self.fm_df['Employees'].mean(),
          'Unemployment': len(self.hh_df[self.hh_df['Employer'].isna()]) / self.hh_df.shape[0],
          }

    self.data.append(df)
    self.no_dealer_found = 0
    self.count += 1
    return 
  

  def get_data(self):
    """
    Type:        Method
    Description: Concatenates data collected after the simulation terminates
    """
    return  pd.DataFrame(self.data)



class Sparse_collector():
  """
  Type:        Class 
  Description: Minimal version of Datacollector. Used in calibraiton.py
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
    # Trade data is not used and thus emptied to save memory
    self.td_data = []
    return

  def get_calibration_data(self):
    """
    Type:        Method
    Description: Concatenates data collected after the simulation terminates
    """
    # Only control households in low saturation areas are used for the calibration process
    # The data is collected only at the last step of the simulation 
    data = [hh.demand for hh in self.model.all_agents if hh.village.market.saturation == 0 and hh.treated == 0]
    data = random.sample(data, k=2713)    
    return data


