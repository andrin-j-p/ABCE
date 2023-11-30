#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import create_agent_data
from data_collector import Sparse_collector, Datacollector, Validation_collector
import arviz as az
import pandas as pd
import seaborn as sns
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


def compare_line(df, y1, y2):

  sns.lineplot(data=df, x='step', y=y1, label='Treated', color= '#5284C2')
  sns.lineplot(data=df, x='step', y=y2, label='Control', color= '#52C290')
  plt.axvline(x=123, color='red', linestyle='--')

  # Set plot labels and title
  plt.xlabel(f'Step')
  plt.ylabel(f"{y1}")
  plt.title(f"Comparison {y1} vs. {y2}")

  # Show plot
  plt.legend()
  plt.show()


model = Model()
model.datacollector = Validation_collector(model)
model.run_simulation(1500)

df = model.datacollector.get_data()

variables = iter(df.columns[1:-1])
for var in variables:
 compare_line(df, var, next(variables))

#%%
print(df[df['step'] == 122])
print(df[df['step'] == 200])
#%%

# Questions
# OK 1) How does intervention group look like? mostly firm owners? -> mostly employed workers with low productivity
# 2) Why dip in treated after token
# 3) Why demand/income/ money increases more for control
# 4) how does benefit differ firm owner and employees
# 4) Why only token permanent effect 

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

print(df['Unemployment'])

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
model.run_simulation(850)

# get simulated data
df_sm_p1, df_hh_p1, df_fm_p1, df_md_p1, df_td_p1 = model.datacollector.get_calibration_data()

# run the model for another 500 periods to compare the change
model.run_simulation(550)
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