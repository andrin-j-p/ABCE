#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import read_dataframe
from Datacollector import Sparse_collector, Datacollector, Validation_collector
import arviz as az
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms

import random
np.random.seed(0) 
random.seed(0)

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


steps = 1000
model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
df_sm_c, df_hh_c, df_fm_c  = model_c.datacollector.get_data()

np.random.seed(0) 
random.seed(0)

model = Model()
model.datacollector = Validation_collector(model)
model.intervention_handler.control = False
model.run_simulation(steps)
df_sm_t, df_hh_t, df_fm_t = model.datacollector.get_data()

def create_lineplots(df_t, df_c, variables):
        
    # Loop through each variable and plot on a separate subplot
    for i, var in enumerate(variables):
      fig, ax = plt.subplots()

      sns.lineplot(data=df_c, x='step', y=var, label='Control', color='#52C290', ax=ax)
      sns.lineplot(data=df_t, x='step', y=var, label='Treated', color='#2C72A6', ax=ax)
        
      # Add a vertical line at x=52
      ax.axvline(x=52, color='red', linestyle='--')
        
      # Set plot labels and title for the current subplot
      ax.set_xlabel('Step')
      ax.set_ylabel(var)
      ax.set_title(f"Comparison {var} Treatment vs. Control")
        
      # Add legend to the current subplot
      ax.legend()
    
      # Show plot
      plt.show()


variables = df_sm_t.columns[1:]
create_lineplots(df_sm_t, df_sm_c, variables)

data = df_sm_c[df_sm_c['step'] == 52] 


df_t = df_sm_t[df_sm_t['step'] == 142] 
df_c = df_sm_c[df_sm_c['step'] == 142] 

#%%
from linearmodels import PanelOLS

def calculate_ATE(df_c, df_t, var):
  # Calculate the treatment effect (difference in means)
  treatment_effect = df_t[var] - df_c[var]
  treatment_effect = np.array(treatment_effect)
    
  # Calculate the ATE (Average Treatment Effect)
  ATE = treatment_effect.mean()
  
  grouped = df.groupby(cluster_var)
  
  # Step 2: Compute group-wise means and variances
  means = grouped[var].mean()
  variances = grouped[var].var()
  
  # Step 3: Combine variances to compute clustered standard errors
  # Compute the cluster-robust variance estimator
  cluster_variance = variances.mean()
  
  # Compute clustered standard errors
  clustered_SE = np.sqrt(cluster_variance)
  
  return clustered_SE

  # Get summary of results
  print(result)
  return ATE

hh_df_recipient_t = df_hh_t[df_hh_t['Treated'] == 1]
hh_df_non_recipient_t = df_hh_t[df_hh_t['Treated'] == 0]                              
fm_df_recipient_t = df_fm_t[df_fm_t['Treated'] == 1]
fm_df_non_recipient_t = df_fm_t[df_fm_t['Treated'] == 0]

hh_df_recipient_c = df_hh_c[df_hh_c['Treated'] == 1]
hh_df_non_recipient_c = df_hh_c[df_hh_c['Treated'] == 0]                               
fm_df_recipient_c = df_fm_c[df_fm_c['Treated'] == 1]
fm_df_non_recipient_c = df_fm_c[df_fm_c['Treated'] == 0]

#%%
converstion = 52*1.871

print(f"               {'Recipients': >13}{'Nonrecipinets':>13}{'Control':>13}")
print(f"HH expenditure {round(float(df_t['Expenditure_Recipient']-df_c['Expenditure_Recipient'])*converstion, 2): >13}{round(float(df_t['Expenditure_Nonrecipient'] - df_c['Expenditure_Nonrecipient'])*converstion, 2) : >13}{round(float(df_c['Expenditure'])*converstion,2) :>13}")
print(f"HH money       {round(float(df_t['Money_Recipient']-df_c['Money_Recipient'])*1.871, 2): >13}{                   round(float(df_t['Money_Nonrecipient'] - df_c['Money_Nonrecipient'])*1.871,2 ) : >13}{     round(float(df_c['Money'])*1.871,2):>13}")
print(f"HH income      {round(float(df_t['Income_Recipient']-df_c['Income_Recipient'])*converstion, 2) : >13}{         round(float(df_t['Income_Nonrecipient'] - df_c['Income_Nonrecipient'] )*converstion,2): >13}{           round(float(df_c['Income'])*converstion,2) :>13}")
print(f"FM profit      {round(float(df_t['Profit_Recipient']-df_c['Profit_Recipient'])*converstion, 2) : >13}{         round(float(df_t['Profit_Nonrecipient'] - df_c['Profit_Nonrecipient'])*converstion,2) : >13}{           round(float(df_c['Profit'])*converstion,2) :>13}")
print(f"FM assets      {round(float(df_t['Assets_Recipient']-df_c['Assets_Recipient'])*1.871, 2): >13}{                round(float(df_t['Assets_Nonrecipient'] - df_c['Assets_Nonrecipient'])*1.871,2 ) : >13}{     round(float(df_c['Assets'])*1.871,2):>13}")
print(f"FM revenue     {round(float(df_t['Revenue_Recipient']-df_c['Revenue_Recipient'])*converstion, 2): >13}{        round(float(df_t['Revenue_Nonrecipient'] - df_c['Revenue_Nonrecipient'])*converstion,2): >13}{          round(float(df_c['Revenue'])*converstion,2) :>13}")
print(f"FM inventory   {round(float(df_t['Stock_Recipient']-df_c['Stock_Recipient']), 2): >13}{                        round(float(df_t['Stock_Nonrecipient'] - df_c['Stock_Nonrecipient']),2): >13}{                          round(float(df_c['Stock']),2) :>13}")

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

def plot_dist(data, title, limit):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(data, ax=ax, label="Observed Data", rug = True, fill_kwargs={'alpha': 0.7}, rug_kwargs={'space':0.1})
  
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density {title}")
  ax.legend()
  plt.xlim(limit[0], limit[1])

  # Show the plot
  plt.tight_layout()
  plt.show()


def compare_dist(data_o, data_c , title, lim):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(data_o, ax=ax, label="Observed", rug = True, rug_kwargs={'space':0.1}, fill_kwargs={'alpha': 0.7})
  az.plot_dist(data_c, ax=ax, label="Simulated", color='red', rug=True,  rug_kwargs={'space':0.2}, fill_kwargs={'alpha': 0.7})
  
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density Comparision True and Simulated {title}")
  ax.legend()

  # Show the plot
  plt.xlim(lim[0], lim[1])
  plt.show()

# instantiate the model and run it for 1000 burn-in periods
model = Model()
model.datacollector = Sparse_collector(model)
model.run_simulation(360)

# get simulated data
df_sm, df_hh, df_fm, df_md, df_td = model.datacollector.get_calibration_data()

# plot distributions
plot_dist(df_hh['demand'].values, 'Expenditure', (0, 500))
plot_dist(df_hh['money'].values, 'Money', (-50, 1000))
plot_dist(df_hh['income'].values, 'Income', (-50, 300))
plot_dist(df_fm['revenue'].values, 'Revenue', (0, 200))
plot_dist(df_fm['profit'].values, 'Profit', (-200, 200))
plot_dist(df_fm['assets'].values, 'Assets', (-500, 2000))
plot_dist(df_fm['stock'].values, 'Stock', (0, 2000))


data_o = 0.01 * read_dataframe("GE_HHLevel_ECMA.dta", "df")['p2_consumption']/52
data_c = df_hh['demand']
compare_dist(data_c, data_o, 'Expenditure Density', (0,100))

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
