#%%
from ABM import Sugarscepe
import numpy as np
import random 
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

np.random.seed(0) 
random.seed(0)

#def main():

"""
Type:         Main function 
Description:  Runs the Simulation and saves the output data 
"""
def compare_line(df_F, df_E, var):
  sns.lineplot(data=df_F, x='step', y=var, label='Firm', color= '#52C290')
  sns.lineplot(data=df_E, x='step', y=var, label='Empl', color= '#2C72A6')
  plt.axvline(x=52, color='red', linestyle='--')

  # Set plot labels and title
  plt.xlabel(f'Step')
  plt.ylabel(f"{var}")
  plt.title(f"Comparison {var} Treatement vs. Control")

  # Show plot
  plt.legend()
  plt.show()

steps = 400
model = Sugarscepe()
model.intervention_handler.control = True
model.run_simulation(steps)

hh_data, fm_data, md_data, _ = model.datacollector.get_data()

df_F = hh_data[hh_data['owns_firm'].notna()]
df_E = hh_data[hh_data['owns_firm'].isna()]

for var in ['income', 'money', 'demand']:
   compare_line(df_F, df_E, var)


print(F"Firm expend: {np.mean([hh.demand for hh in model.all_agents if hh.firm != None])}")
print(F"Empl expend: {np.mean([hh.demand for hh in model.all_agents if hh.firm == None])}")

print(F"Firm money: {np.mean([hh.money for hh in model.all_agents if hh.firm != None])}")
print(F"Empl money: {np.mean([hh.money for hh in model.all_agents if hh.firm == None])}")

print(F"Firm income: {np.mean([hh.income for hh in model.all_agents if hh.firm != None])}")
print(F"Empl income: {np.mean([hh.income for hh in model.all_agents if hh.firm == None])}")

#%%

#%%
# Pickle the DataFrame
with open('../data/output_data/model_output.pkl', 'wb') as file:
    pickle.dump((hh_data, fm_data, md_data), file)

# Open the pickled file and load the DataFrame
with open('../data/output_data/model_output.pkl', 'rb') as file:
    hh_data, fm_data, md_data = pickle.load(file)
  

#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import create_agent_data
from Datacollector import Sparse_collector, Datacollector, Validation_collector
import arviz as az
import pandas as pd
import seaborn as sns
import random
np.random.seed(0) 
random.seed(0)
#@ TODO 
# Make plots prettier:
# - work out color concept
# - fonts 
# - etc.
# Add confidence intervalls to line plot

# @DELETE set display options. Not imperative for exectution
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)


class Model(ABM.Sugarscepe):
  """
  Type:        Child Class Sugarscepe
  Description: Implements the following additional functionality imperative for the calibration process:
              - sparse data collecter for faster execution
  """
  def __init__(self):
    super().__init__()
    self.datacollector = Sparse_collector(self) #Validation_collector(self)#
    self.data = []

def compare_line(df_t, df_c, var):
  sns.lineplot(data=df_c, x='step', y=var, label='Control', color= '#52C290')
  sns.lineplot(data=df_t, x='step', y=var, label='Treated', color= '#2C72A6')
  plt.axvline(x=52, color='red', linestyle='--')
  # Set plot labels and title
  plt.xlabel(f'Step')
  plt.ylabel(f"{var}")
  plt.title(f"Comparison {var} Treatement vs. Control")
  # Show plot
  plt.legend()
  plt.show()



def batch_runner(hh_attr):

  steps = 1000


  model_c = Model()
  model_c.datacollector = Validation_collector(model_c)
  model_c.intervention_handler.control = True
    
  for hh in model_c.all_agents:
    for parameter, value in hh_attr.items():
      setattr(hh, parameter, value)


  model_c.run_simulation(steps)
  df_c_1 = model_c.datacollector.get_data()


###############################################

  np.random.seed(0) 
  random.seed(0)

  model = Model()
  model.datacollector = Validation_collector(model)
  model.intervention_handler.control = False

    
  for hh in model.all_agents:
    for parameter, value in hh_attr.items():
      setattr(hh, parameter, value)

  model.run_simulation(steps)
  df_t_1 = model.datacollector.get_data()


  variables_t = iter(df_t_1.columns[1:-1])

  for var in variables_t:
   compare_line(df_t_1, df_c_1, var)


  df_t = df_t_1[df_t_1['step'] == 142] 
  df_c = df_c_1[df_c_1['step'] == 142] 

  print(hh_attr)
  converstion = 52*1.871
  print(f"               {'Recipients': >13}{'Nonrecipinets':>13}{'Control':>13}")
  print(f"HH expenditure {round(float(df_t['Expenditure_Recipient']-df_c['Expenditure_Recipient'])*converstion, 2): >13}{round(float(df_t['Expenditure_Nonrecipient'] - df_c['Expenditure_Nonrecipient'])*converstion, 2) : >13}{round(float(df_c['Expenditure_Nonrecipient'])*converstion,2) :>13}")
  print(f"HH money      {round(float(df_t['Money_Recipient']-df_c['Money_Recipient'])*1.871, 2): >13}{                     round(float(df_t['Money_Nonrecipient'] - df_c['Money_Nonrecipient'])*1.871,2 ) : >13}{     round(float(df_c['Assets_Nonrecipient'])*1.871,2):>13}")
  print(f"HH income      {round(float(df_t['Income_Recipient']-df_c['Income_Recipient'])*converstion, 2) : >13}{        round(float(df_t['Income_Nonrecipient'] - df_c['Income_Nonrecipient'] )*converstion,2): >13}{           round(float(df_c['Income_Nonrecipient'])*converstion,2) :>13}")
  print(f"FM profit      {round(float(df_t['Profit_Recipient']-df_c['Profit_Recipient'])*converstion, 2) : >13}{        round(float(df_t['Profit_Nonrecipient'] - df_c['Profit_Nonrecipient'])*converstion,2) : >13}{           round(float(df_c['Profit_Nonrecipient'])*converstion,2) :>13}")
  print(f"FM assets      {round(float(df_t['Assets_Recipient']-df_c['Assets_Recipient'])*1.871, 2): >13}{                     round(float(df_t['Assets_Nonrecipient'] - df_c['Expenditure_Nonrecipient'])*1.871,2 ) : >13}{     round(float(df_c['Assets_Nonrecipient'])*1.871,2):>13}")
  print(f"FM revenue     {round(float(df_t['Revenue_Recipient']-df_c['Revenue_Recipient'])*converstion, 2): >13}{       round(float(df_t['Revenue_Nonrecipient'] - df_c['Revenue_Nonrecipient'])*converstion,2): >13}{          round(float(df_c['Revenue_Nonrecipient'])*converstion,2) :>13}")
  print(f"FM inventory   {round(float(df_t['Stock_Recipient']-df_c['Stock_Recipient']), 2): >13}{                       round(float(df_t['Stock_Nonrecipient'] - df_c['Stock_Nonrecipient']),2): >13}{                          round(float(df_c['Stock_Nonrecipient']),2) :>13}")

  print(f"Expenditure Recipient: {converstion* np.mean([hh.demand for hh in model.all_agents if hh.treated == 1])}")
  print(f"Expenditure Non recipient: {converstion* np.mean([hh.demand for hh in model.all_agents if hh.treated == 0])}")

  print(f"Expenditure Firm: {converstion* np.mean([hh.demand for hh in model.all_agents if hh.firm != None])}")
  print(f"Expenditure No Firm: {converstion* np.mean([hh.demand for hh in model.all_agents if hh.firm == None])}")
  print(f"Unemployment T: {df_t['Unemployment']}")
  print(f"Unemployment C: {df_c['Unemployment']}")
  print(F"Price T: {np.mean([fm.price for fm in model.all_firms])}")
  print(F"Price C: {np.mean([fm.price for fm in model_c.all_firms])}")

  print(F"Productivity R: {np.mean([hh.productivity for hh in model.all_agents if hh.treated == 1])}")
  print(F"Productivity NR: {np.mean([hh.productivity for hh in model.all_agents if hh.treated == 0])}")

  print(F"Firm R: {np.mean([1 if hh.treated == 1 and hh.firm!= None else 0 for hh in model.all_agents ])}")
  print(F"Firm NR: {np.mean([1 if hh.treated == 0 and hh.firm!= None else 0 for hh in model.all_agents ])}")


hh_attrs = [{'alpha': 0.78}, {'alpha': 0.79}, {'alpha': 0.8},] 

for hh_attr in  hh_attrs:
  batch_runner(hh_attr=hh_attr)


if __name__ == "__main__":

  print('')
# %%
