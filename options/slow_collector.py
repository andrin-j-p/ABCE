class Datacollector():
  """
  Type:        Helper Class 
  Description: Collects data at the agent and model level
  """
  def __init__(self, model):
    self.model = model
    self.hh_data = []
    self.sm_data = []
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

### Agent Dataframe
    agent_data = [(step, agent.unique_id, agent.village.unique_id, agent.pos[0], agent.pos[1], agent.income, agent.best_dealers,
                   agent.consumption, agent.money, agent.demand, agent.employer, agent.firm)
                  for agent in self.model.all_agents]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_hh = pd.DataFrame(agent_data, columns=['step', 'unique_id', 'village_id', 'lat', 'lon', "income", "best_dealers", "consumption", "money", 
                                              'demand', 'employer', 'owns_firm'])
    
    self.hh_data.append(df_hh[['step', 'lat', 'lon', 'income', 'money']])
    
### Firm Dataframe
    firm_data = [(firm.unique_id, firm.stock, firm.price, firm.money, firm.output, firm.sales, firm.price * firm.sales,
                  len(firm.employees), len(set(firm.costumers)))
                 for firm in self.model.all_firms]
    

    df_fm = pd.DataFrame(firm_data, columns=['unique_id', 'stock', 'price', 'money', 'output', 'sales', 'revenue', 'employees', 
                                             'costumers'])
 
### Trade Dataframe
    df_td = pd.DataFrame(self.td_data)

### Summarize data
    sm_data = [(step,  
                df_fm['output'].mean(),
                df_fm['employees'].mean(), 
                df_fm['stock'].mean(), 
                df_td['amount'].sum(), 
                (sum(1 for item in df_hh['employer']  if item == None) / len(self.model.all_agents)),
                df_hh['income'].mean(),
                df_td['price'].mean(), 
                df_td['volume'].sum(),
                (df_hh['demand'].sum()/7),
                df_fm['output'].sum(),
                self.no_dealer_found, 
                self.no_worker_found,
                self.worker_fired)
                for step in range(self.model.schedule.steps + 1)]
    
    # Create a Pandas DataFrames from the list comprehensions
    df_sm = pd.DataFrame(sm_data, columns=['step','total_output', 'average_employees', 'average_stock', 'total_sales', 'unemployment_rate',
                                            'average_income', 'average_price', 'trade_volume', 'demand_satisfied', 'output', 
                                            'no_dealer_found', 'no_worker_found', 'worker_fired'])
    
    # Put all model level data into one dataframe
    self.no_dealer_found = 0
    self.no_worker_found = 0
    self.worker_fired = 0

    self.sm_data.append(df_sm)
    

  
  def get_data(self):

    df_sm = pd.concat(self.sm_data, axis=0)
    df_hh = pd.concat(self.hh_data, axis=0)

    return df_hh, df_sm

