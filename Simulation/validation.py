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


steps = 1000
model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
df_c_1 = model_c.datacollector.get_data()

np.random.seed(0) 
random.seed(0)

model = Model()
model.datacollector = Validation_collector(model)
model.intervention_handler.control = False
model.run_simulation(steps)
df_t_1 = model.datacollector.get_data()


variables_t = iter(df_t_1.columns[1:-1])

for var in variables_t:
 compare_line(df_t_1, df_c_1, var)


df_t = df_t_1[df_t_1['step'] == 142] 
df_c = df_c_1[df_c_1['step'] == 142] 

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



#%%

def compare_dist(p1, p2, title, lim):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(p1, ax=ax, label="Treated", rug = True, rug_kwargs={'space':0.1}, fill_kwargs={'alpha': 0.7})
  az.plot_dist(p2, ax=ax, label="Control", color='red', rug=True,  rug_kwargs={'space':0.2}, fill_kwargs={'alpha': 0.7})
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density Comparision True and Simulated {title}")
  ax.legend()

  # Show the plot
  plt.xlim(lim[0], lim[1])
  plt.show()

print(df_t_1['Expenditure_Recipient'].values)
compare_dist(df_t_1['Expenditure_Recipient'].values, df_c_1['Expenditure_Recipient'], 'Expenditure_Recipient', (0, 100) )
compare_dist(df_t_1['Expenditure_Nonrecipient'].values, df_c_1['Expenditure_Nonrecipient'], 'Expenditure_Nonrecipient', (0, 100) )

#%%
steps = 350
model = Model()
model.datacollector = Sparse_collector(model)
model.run_simulation(steps)

print(f"Money_1 all hh {np.mean([hh.money for hh in model.all_agents])}")
print(f"Money_1 firm {np.mean([hh.money for hh in model.all_agents if hh.firm != None])}")
print(f"Money_1 no firm {np.mean([hh.money for hh in model.all_agents if hh.firm == None])}")

model.run_simulation(650)
print(f"Money_2 all hh {np.mean([hh.money for hh in model.all_agents])}")
print(f"Money_2 firm {np.mean([hh.money for hh in model.all_agents if hh.firm != None])}")
print(f"Money_2 no firm {np.mean([hh.money for hh in model.all_agents if hh.firm == None])}")

#%%%
# Questions
# OK 1) How does intervention group look like? mostly firm owners? -> mostly employed workers with low productivity
# 2) Why dip in treated after token
# 3) Why demand/income/ money increases more for control
# 4) how does benefit differ firm owner and employees
# OK 4) Why only token permanent effect? It does not, it is due to split into rich and poor 

# Explenations
# 2) got unemployed recently not yet at rock bottom (plausible if treated mostly workers)
# 2) firm goes through rough patch (plausible if treated mostly firm owners)
# 4) only so many more worker they can hire  

# Approaches
# 0) Scatter the distribution of money to make it last longer
# 0) More firms -> more firm owners in treatment check how in data
# 0) Choose 50% poorest and then randomize (ansers Q4? )
# 0) Make hh act on different markets
# 1) wage depending on firm performance
# 1) Investment how? If firm assets increase make cobb douglas with capital
# 2) Make effect last longer f.i. propensity to consume
# 2) Alternative ocupation how? and how useful?
# 2) Incorporate slack how?

# Insights
# 1) In simulation most recipients are employed low productivity households. Might explain why effect on other hh is just as much 

print(df_t['Unemployment'])

print(f" Treated firm: {len([agent for agent in model.treated_agents if agent.firm != None])}")
print(f" Treaded no firm: {len([agent for agent in model.treated_agents if agent.firm == None])}")
print(f" Treaded empleyd: {len([agent for agent in model.treated_agents if agent.employer != None])}")
print(f" Treaded unemployed: {len([agent for agent in model.treated_agents if  agent.employer == None])}")
print(f" Treaded productivity: {np.mean([agent.productivity for agent in model.treated_agents])}")
print(f" Total average productivity: {np.mean([agent.productivity for agent in model.all_agents])}")

#%%
def compare_dist(p1, p2, title, lim):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(p1, ax=ax, label="Observed Data", rug = True, rug_kwargs={'space':0.1}, fill_kwargs={'alpha': 0.7})
  az.plot_dist(p2, ax=ax, label="Simulated Data", color='red', rug=True,  rug_kwargs={'space':0.2}, fill_kwargs={'alpha': 0.7})
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density Comparision True and Simulated {title}")
  ax.legend()

  # Show the plot
  plt.xlim(lim[0], lim[1])
  plt.show()


def plot_dist(data, title, lim):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(data, ax=ax, label="Observed Data", rug = True, fill_kwargs={'alpha': 0.7}, rug_kwargs={'space':0.1})
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density {title}")
  ax.legend()

  # Show the plot
  plt.xlim(lim[0], lim[1])
  plt.show()


# instantiate the model and run it for 1000 burn-in periods
model = Model()
model.datacollector = Sparse_collector(model)
model.run_simulation(800)

# get simulated data
df_sm_p1, df_hh_p1, df_fm_p1, df_md_p1, df_td_p1 = model.datacollector.get_calibration_data()

# run the model for another 500 periods to compare the change
model.run_simulation(20)
df_sm_p2, df_hh_p2, df_fm_p2, df_md_p2, df_td_p2 = model.datacollector.get_calibration_data() 


# compare distributions
compare_dist(df_hh_p1['demand'].values, df_hh_p2['demand'], 'Demand', (0, 70) )
compare_dist(df_hh_p1['money'].values, df_hh_p2['money'], 'Money', (0, 700) )
compare_dist(df_fm_p1['profit'].values, df_fm_p2['profit'], 'Profit', (-200, 700) )
compare_dist(df_fm_p1['assets'].values, df_fm_p2['assets'], 'Assets', (-1000, 2000) )
compare_dist(df_fm_p1['stock'].values, df_fm_p2['stock'].values, 'Stock', (0,500))

# get summary statistics
print(f"p1 vs. p2 income mean: {np.mean(df_hh_p1['income'])} | {np.mean(df_hh_p2['income'])}")
print(f"p1 vs. p2 consumption mean: { np.mean(df_hh_p1['demand'].values)} | { np.mean(df_hh_p2['demand'].values)}")
print(f"p1 vs. p2 profit: {np.mean(df_fm_p1['profit'])} | {np.mean(df_fm_p2['profit'])}")
print(f"p1 vs. p2 unmeployment rate: {sum(1 for row in df_hh_p1.itertuples()  if row.employer == None) / df_hh_p1.shape[0]} | {sum(1 for row in df_hh_p2.itertuples()  if row.employer == None) / df_hh_p1.shape[0]}")
print(f"p1 vs. p2 total zero income: {sum(1 for row in df_hh_p1.itertuples()  if row.income == 0)} | {sum(1 for row in df_hh_p2.itertuples()  if row.income == 0)}")
print(f"zero income no firm: {sum(1 for row in df_hh_p1.itertuples()  if row.income == 0 and row.firm == None)}")
print(f"zero income owns firm: {sum(1 for row in df_hh_p1.itertuples()  if row.income == 0 and row.firm != None)}")
print(f"total negative assets: {sum(row.assets for row in df_fm_p1.itertuples()  if row.assets < 0)}")
print(f" negative assets: {len(pd.DataFrame([row for row in df_fm_p1.itertuples() if row.assets <= 0 ]))}")
print(f" negative profit {len(pd.DataFrame([row for row in df_fm_p1.itertuples() if row.profit <= 0]))} ")

# plot distributions
plot_dist(df_fm_p1['profit'].values, 'simulated profit', (-200, 200))
plot_dist(df_fm_p1['assets'].values, 'fm assets', (-2000, 2000))
plot_dist(df_fm_p1['stock'].values, 'fm stock', (-300, 2000))
plot_dist(df_hh_p1['income'].values, 'hh income', (-50, 300))
plot_dist(df_hh_p1[df_hh_p1['firm'].notna()] ['income'], 'hh income firm owners',  (-50, 300))
plot_dist(df_hh_p1['money'].values, 'hh money', (-50, 1000))
plot_dist(df_hh_p1[df_hh_p1['firm'].notna()] ['money'], 'hh money firm owners',  (-50, 1000))
#%%
# compare distribuiton 
df_hh_true['p3_totincome'] = 0.01 * df_hh_true['p3_totincome']/52
df_hh_true['p2_consumption'] = 0.01 * df_hh_true['p2_consumption']/52
df_fm_true['prof_year'] = 0.01 * df_fm_true['prof_year']
df_fm_true['rev_year'] = 0.01 * df_fm_true['rev_year']