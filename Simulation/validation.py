#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import create_agent_data
from data_collector import Sparse_collector
import arviz as az
import pandas as pd


# @DELETE set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)

class Model(ABM.Sugarscepe):
  """
  Type:        Child Class Sugarscepe
  Description: Implements the following additional functionality imperative for the calibration process:
  """
  def __init__(self):
    super().__init__()
    self.datacollector = Sparse_collector(self)
    self.data = []


  
def compare_dist(true, simulated, title, lim):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(true, ax=ax, label="Observed Data", rug = True, quantiles=[0.05, 0.5, 0.95], rug_kwargs={'space':0.1})
  az.plot_dist(simulated, ax=ax, label="Simulated Data", color='red', rug=True, quantiles=[0.05, 0.5, 0.95], rug_kwargs={'space':0.2}, fill_kwargs={'alpha': 0.7})
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
  az.plot_dist(data, ax=ax, label="Observed Data", rug = True, quantiles=[0.05, 0.5, 0.95], rug_kwargs={'space':0.1})
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density {title}")
  ax.legend()

  # Show the plot
  plt.xlim(lim[0], lim[1])
  plt.show()

# instantiate the model and run it for N periods
model = Model()
model.run_simulation(300)

# get simulated data
df_sm_sim, df_hh_sim, df_fm_sim = model.datacollector.get_calibration_data()

# get observed data
df_hh_true, df_fm_true, _, _ = create_agent_data()

# compare distribuiton 
df_hh_true['p3_totincome'] = 0.01 * df_hh_true['p3_totincome']/52
df_hh_true['p2_consumption'] = 0.01 * df_hh_true['p2_consumption']/52
df_fm_true['prof_year'] = 0.01 * df_fm_true['prof_year']
df_fm_true['rev_year'] = 0.01 * df_fm_true['rev_year']

compare_dist(df_hh_true['p2_consumption'].values, df_hh_sim['demand'].values, 'Demand', (0, 50))
#plot_dist(df_hh_true['p3_totincome'].values, 'true income')
#plot_dist(df_fm_sim['profit'].values, 'simulated profit')
#plot_dist(df_fm_true['prof_year'].values, 'true profits')
#compare_dist(df_fm_true['prof_year'].values, df_fm_sim['profit'].values, 'Profit', (-800, 2000))
plot_dist(df_fm_sim['profit'].values, 'simulated profit', (-200, 200))
plot_dist(df_fm_sim['assets'].values, 'fm assets', (-1000, 1000))

plot_dist(df_hh_sim['income'].values, 'hh income', (-100, 200))
plot_dist(df_hh_sim['money'].values, 'hh money', (-5000, 4000))

#==========================================================================
print(f"true vs. simulated income mean: {np.mean(df_hh_true['p3_totincome']), np.mean(df_hh_sim['income'])}")
print(f"true vs. simulated income median: {np.median(df_hh_true['p3_totincome']), np.median(df_hh_sim['income'])}")

print(f"true vs simulated consumption mean: {np.median(df_hh_true['p2_consumption'].dropna().values), np.mean(df_hh_sim['demand'].values)}")
print(f"true vs simulated consumption median: {np.mean(df_hh_true['p2_consumption'].dropna().values), np.mean(df_hh_sim['demand'].values)}")

print(f"simulated average profit{np.mean(df_fm_sim['profit'])}")

(sum(1 for item in df_hh_sim['employer']  if item == None) / df_hh_sim.shape[0]),

# %%
print(pd.DataFrame([row for row in df_hh_sim.itertuples() if row.money < 0 ]))

#print(pd.DataFrame([row for row in df_fm_sim.itertuples() if row.costumers == 0 ]))
# %%
