#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import create_agent_data
from data_collector import Sparse_collector
import arviz as az


class Model(ABM.Sugarscepe):
  """
  Type:        Child Class Sugarscepe
  Description: Implements the following additional functionality imperative for the calibration process:
  """
  def __init__(self):
    super().__init__()
    self.datacollector = Sparse_collector(self)
    self.data = []
  
  def run_simulation(self, steps):
    """
    Type:        Method
    Description: Overwrite run simulaiton in parent class to avoid print statement
    """
    for i in range(steps):
      self.step()

  
def compare_dist(true, simulated):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(true, ax=ax, label="Observed Data", rug = True, quantiles=[0.05, 0.5, 0.95], rug_kwargs={'space':0.1})
  az.plot_dist(simulated, ax=ax, label="Simulated Data", rug=True, quantiles=[0.05, 0.5, 0.95], rug_kwargs={'space':0.2}, fill_kwargs={'alpha': 0.7})
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title("Kernel Density Comparision True and Simulated")
  ax.legend()

  # Show the plot
  plt.xlim(0, 200)
  plt.show()

# instantiate the model and run it for N periods
model = Model()
model.run_simulation(190)

# get simulated data
df_sm_sim, df_hh_sim, df_fm_sim = model.datacollector.get_calibration_data()

# get observed data
df_hh_true, df_fm_true, _, _ = create_agent_data()

# compare distribuiton 
compare_dist(0.01 * df_hh_true['p2_consumption'].values/52, df_hh_sim['consumption'].values)
#compare_dist(0.01 * df_hh_true['p3_totincome'].values/52, df_hh_sim['income'].values)
# %%
print(df_hh_sim['consumption'].values)
print(f"true mean: {0.01 * np.median(df_hh_true['p2_consumption'].dropna().values/52)}")
print(f"simulated mean: {np.median(df_hh_sim['consumption'].values)}")
# %%
