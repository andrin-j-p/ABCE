#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import create_agent_data
from data_collector import Sparse_collector, Datacollector
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
              - sparse data collecter for faster execution
  """
  def __init__(self):
    super().__init__()
    self.datacollector = Datacollector(self)#Sparse_collector(self)
    self.data = []


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

def validation1():
  # instantiate the model and run it for 1000 burn-in periods
  model = Model()
  model.run_simulation(800)

  # get simulated data
  df_sm_p1, df_hh_p1, df_fm_p1, df_md_p1, df_td_p1 = model.datacollector.get_calibration_data()

  # run the model for another 500 periods to compare the change
  model.run_simulation(497)
  df_sm_p2, df_hh_p2, df_fm_p2, df_md_p2, df_td_p2 = model.datacollector.get_calibration_data() 

  # get observed data
  df_hh_true, df_fm_true, _, _ = create_agent_data()

  compare_dist(df_hh_p1['demand'].values, df_hh_p2['demand'], 'Demand', (0, 70) )
  compare_dist(df_hh_p1['money'].values, df_hh_p2['money'], 'Money', (0, 700) )
  compare_dist(df_fm_p1['profit'].values, df_fm_p2['profit'], 'Profit', (-200, 700) )
  compare_dist(df_fm_p1['assets'].values, df_fm_p2['assets'], 'Assets', (-1000, 2000) )
  compare_dist(df_fm_p1['stock'].values, df_fm_p2['stock'].values, 'Stock', (0,500))

def scatter(x, y, title=''):

  # Scatter plot
  plt.scatter(x, y)

  # Add labels and title
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title(f"Scatter Plot of {title}")

  # Display the plot
  plt.show()
            
def validation2():
  model = Model()
  model.run_simulation(1300)
  hh_data, fm_data, md_data, _ = model.datacollector.get_data()
  scatter(md_data['step'], md_data['average_profit'], 'Profit')
  scatter(md_data['step'], md_data['average_income'], 'Income')



validation2()


#%%
#==========================================================================
print(f"p1 vs. p2 income mean: {np.mean(df_hh_p1['income'])} | {np.mean(df_hh_p2['income'])}")
print(f"p1 vs. p2 consumption mean: { np.mean(df_hh_p1['demand'].values)} | { np.mean(df_hh_p2['demand'].values)}")
print(f"p1 vs. p2 profit: {np.mean(df_fm_p1['profit'])} | {np.mean(df_fm_p2['profit'])}")
print(f"p1 vs. p2 unmeployment rate: {sum(1 for row in df_hh_p1.itertuples()  if row.employer == None) / df_hh_p1.shape[0]} | {sum(1 for row in df_hh_p2.itertuples()  if row.employer == None) / df_hh_p1.shape[0]}")
print(f"p1 vs. p2 total zero income: {sum(1 for row in df_hh_p1.itertuples()  if row.income == 0)} | {sum(1 for row in df_hh_p2.itertuples()  if row.income == 0)}")

#%%

# compare distribuiton 
df_hh_true['p3_totincome'] = 0.01 * df_hh_true['p3_totincome']/52
df_hh_true['p2_consumption'] = 0.01 * df_hh_true['p2_consumption']/52
df_fm_true['prof_year'] = 0.01 * df_fm_true['prof_year']
df_fm_true['rev_year'] = 0.01 * df_fm_true['rev_year']

compare_dist(df_hh_true['p2_consumption'].values, df_hh_p1['demand'].values, 'Demand', (-1, 70))
plot_dist(df_fm_p1['profit'].values, 'simulated profit', (-200, 200))
plot_dist(df_fm_p1['assets'].values, 'fm assets', (-2000, 2000))
plot_dist(df_fm_p1['stock'].values, 'fm stock', (-300, 2000))
plot_dist(df_hh_p1['income'].values, 'hh income', (-50, 300))
plot_dist(df_hh_p1[df_hh_p1['firm'].notna()] ['income'], 'hh income firm owners',  (-50, 300))
plot_dist(df_hh_p1['money'].values, 'hh money', (-50, 1000))
plot_dist(df_hh_p1[df_hh_p1['firm'].notna()] ['money'], 'hh money firm owners',  (-50, 1000))

print(f"zero income no firm: {sum(1 for row in df_hh_p1.itertuples()  if row.income == 0 and row.firm == None)}")
print(f"zero income owns firm: {sum(1 for row in df_hh_p1.itertuples()  if row.income == 0 and row.firm != None)}")
print(f"total negative assets: {sum(row.assets for row in df_fm_p1.itertuples()  if row.assets < 0)}")
print(f" negative assets: {len(pd.DataFrame([row for row in df_fm_p1.itertuples() if row.assets <= 0 ]))}")
print(f" negative profit {len(pd.DataFrame([row for row in df_fm_p1.itertuples() if row.profit <= 0]))} ")
print(df_md_p1.tail(100))

#%%
print(pd.DataFrame([row for row in df_fm_p1.itertuples() if row.assets < 0 ]))
print(f" negative profit {len(pd.DataFrame([row for row in df_fm_p1.itertuples() if row.profit <= 0]))} ")


#%%

# @DELETE set display options. Not imperative for exectution
pd.set_option('display.max_columns', 20)
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
    self.datacollector = Datacollector(self)
    self.data = []

steps = 200
model = Model()
model.run_simulation(steps)
hh_data, fm_data, md_data, _ = model.datacollector.get_data()
print(fm_data[(fm_data['step'] == steps-1) & (fm_data['assets'] < 0)])
# %%

print(fm_data[fm_data['unique_id'] == 'f_601010101004-013'])
# %%
