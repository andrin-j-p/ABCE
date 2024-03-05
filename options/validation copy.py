#%%
import numpy as np
from matplotlib import pyplot as plt
import ABM
from read_data import read_dataframe
from datacollector import Sparse_collector, Validation_collector
import arviz as az
import pandas as pd
import seaborn as sns
import random
np.random.seed(0) 
random.seed(0)

#@DELETE
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


#%%
"""
State of the economy before intervention:
- Expenditure
- Income
- Assets
- Profit
- Revenue

- Firm size
- 
"""    
steps = 360
model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
_, data_hh_c, data_fm_c  = model_c.datacollector.get_data()


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


def compare_histograms(data_o, data_c):
  bins = np.arange(1, 17)
  density_o, _ = np.histogram(data_o, bins=bins, density=True)
  density_c, _ = np.histogram(data_c, bins=bins, density=True)

  fig, ax = plt.subplots()  
  pos = np.arange(15)
  wid = 0.3
  plt.bar(pos, density_o, width=wid, color=(0.1, 0.2, 0.5) , label='Observed')
  plt.bar(pos + wid, density_c, width=wid, color=(0.8, 0.1, 0.1), label='Simulated')

  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Number of Firm Employees")
  ax.legend()
  plt.xticks(pos + wid / 2, [i for i in range(1,16)])
  plt.show()

data_hh_o = read_dataframe("GE_HH-Survey-BL_Analysis_AllHHs.dta", "df")
#data_hh_o = data_hh_o[(data_hh_o['hi_sat']==0) & (data_hh_o['treat'] == 0)]

data_fm_o = read_dataframe("GE_ENT-Survey-BL_Analysis_ECMA.dta", "df")
#data_fm_o = data_hh_o[(data_hh_o['hi_sat']==0) & (data_hh_o['treat'] == 0)]

df_hh_ = model_c.datacollector.hh_df
#data_c = df_hh[(df_hh['Saturation'] == 0) & (df_hh['Treated'] == 0)]

#Expenditure
#expenditure_o = data_hh_o['p2_consumption_PPP']/10
expenditure_c = data_hh_c['Expenditure']
#compare_dist(expenditure_o, expenditure_c, 'Expenditure Density', (0,100))

#Money 
money_o = data_hh_o['p1_assets_trim_PPP_BL']/10
money_c = data_hh_c['Money']
#compare_dist(money_o, money_c, 'Money Density', (0,1000))

#Wage Income
income_o = data_hh_o['total_income_trim']/520*1.872
income_c = data_hh_c['Income']
#compare_dist(income_o, income_c, 'Income Density', (0,50))

data_hh_o.replace(0.00, np.nan, inplace=True)  
data_hh_o.dropna(subset=['p3_2_nonagprofit_PPP_BL', 'p4_totrevenue_trim_BL'], inplace=True)  

#compare_dist(profit_o, profit_c, 'Profit Density', (1,10))
plot_dist(data_fm_c['Profit'], 'Profit', (0,50))
plot_dist(data_fm_c['Revenue'], 'Revenue', (0,200))

plot_dist(data_hh_o['p3_2_nonagprofit_PPP_BL']/52, 'Profit', (0,50))
plot_dist(data_hh_o['p4_totrevenue_trim_BL']/520 *1.871, 'Revenue', (0,500))

#compare_dist(data_hh_o['p3_2_nonagprofit_PPP_BL'], data_fm_c['Profit'], 'Revenue Density', (0,10))
#compare_dist(ratio_o, ratio_c, 'Revenue Density', (0,1))

#Revenue
revenue_o = data_hh_o['p4_totrevenue_trim_BL']/52 *1.871
revenue_c = data_fm_c['Revenue']
#compare_dist(revenue_o, revenue_c, 'Revenue Density', (0,300))
#plot_dist(revenue_o, 'Revenue', (0,300))


# Number employees (firm size)
df = read_dataframe("GE_Enterprise_ECMA.dta", "df")

df = read_dataframe("GE_Enterprise_ECMA.dta", "df")
df.dropna(subset=['emp_n_tot'], inplace=True)
df = df[df['emp_n_tot'].apply(lambda x: 0 < x <= 15)]
employees_o = df['emp_n_tot']

df_c = data_fm_c[data_fm_c['Employees'].apply(lambda x: 0 < x <= 15)]
employees_c = df_c['Employees']

compare_histograms(employees_o, employees_c)



#%%
steps = 1000

np.random.seed(0) 
random.seed(0)

model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
df_sm_c, df_hh_c, df_fm_c  = model_c.datacollector.get_data()

np.random.seed(0) 
random.seed(0)

model_t = Model()
model_t.datacollector = Validation_collector(model_t)
model_t.intervention_handler.control = False
model_t.run_simulation(steps)
df_sm_t, df_hh_t, df_fm_t = model_t.datacollector.get_data()


#%%
"""
Calculates main intervention outcomes:
- expenditure
- Money
- Income
- Revenue
- Profit
- Stock 
- Wage
- Unemployment

"""
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

df_sm_t['Profit Margin'] = df_sm_t['Profit'] / df_sm_t['Revenue']
df_sm_c['Profit Margin'] = df_sm_c['Profit'] / df_sm_c['Revenue']
variables = df_sm_t.columns[1:]

create_lineplots(df_sm_t, df_sm_c, variables)
#%%


def create_lineplots(df_t, df_c):

    variables = ['Expenditure', 'Money', 'Income', 'Revenue', 'Profit', 'Unemployment']

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(25, 10))  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Variables to store legend handles and labels
    legend_handles, legend_labels = [], []

    for i, var in enumerate(variables):
        ax = axes[i]

        # Plot control and treated data on the current subplot axis
        sns.lineplot(data=df_c, x='step', y=var, label='Counterfactual', color='#52C290', ax=ax, legend=False)
        sns.lineplot(data=df_t, x='step', y=var, label='Intervention', color='#2C72A6', ax=ax, legend=False)

        # Add a vertical line at x=52
        ax.axvline(x=52, color='red', linestyle='--')

        # Set plot labels and title
        ax.set_xlabel('Week')
        ax.set_ylabel(var)
        ax.set_title(f"{var}")

        # If it's the first iteration, capture the legend handles and labels
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels)

    # Create a single legend for the entire figure using the handles and labels captured
    fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1.1, 1), title='')

    # Adjust layout for better spacing
    fig.subplots_adjust(right=0.5) 
    plt.tight_layout()  # Adjust the rect if legend overlaps with subplots

    # Save the figure
    plt.savefig('../data/illustrations/lineplots.jpg', format='jpg', dpi=1000) 

    # Show plot
    plt.show()



create_lineplots(df_sm_t, df_sm_c)

df_t = df_sm_t[df_sm_t['step'] == 142] 
df_c = df_sm_c[df_sm_c['step'] == 142] 

conversion  = 52
print(f"               {'Recipients': >13}{'Nonrecipinets':>13}{'Control':>13}")
print(f"HH expenditure {round(float(df_t['Expenditure_Recipient']-df_c['Expenditure_Recipient'])*conversion , 2): >13}{round(float(df_t['Expenditure_Nonrecipient'] - df_c['Expenditure_Nonrecipient'])*conversion , 2) : >13}{round(float(df_c['Expenditure'])*conversion ,2) :>13}")
print(f"HH money       {round(float(df_t['Money_Recipient']-df_c['Money_Recipient']), 2): >13}{                        round(float(df_t['Money_Nonrecipient'] - df_c['Money_Nonrecipient']), 2 ) : >13}{                       round(float(df_c['Money']),2):>13}")
print(f"HH income      {round(float(df_t['Income_Recipient']-df_c['Income_Recipient'])*conversion , 2) : >13}{         round(float(df_t['Income_Nonrecipient'] - df_c['Income_Nonrecipient'] )*conversion ,2): >13}{           round(float(df_c['Income'])*conversion ,2) :>13}")
print(f"FM profit      {round(float(df_t['Profit_Recipient']-df_c['Profit_Recipient'])*conversion , 2) : >13}{         round(float(df_t['Profit_Nonrecipient'] - df_c['Profit_Nonrecipient'])*conversion ,2) : >13}{           round(float(df_c['Profit'])*conversion ,2) :>13}")
print(f"FM assets      {round(float(df_t['Assets_Recipient']-df_c['Assets_Recipient']), 2): >13}{                      round(float(df_t['Assets_Nonrecipient'] - df_c['Assets_Nonrecipient']),2 ) : >13}{                      round(float(df_c['Assets']),2):>13}")
print(f"FM revenue     {round(float(df_t['Revenue_Recipient']-df_c['Revenue_Recipient'])*conversion , 2): >13}{        round(float(df_t['Revenue_Nonrecipient'] - df_c['Revenue_Nonrecipient'])*conversion ,2): >13}{          round(float(df_c['Revenue'])*conversion ,2) :>13}")
print(f"FM inventory   {round(float(df_t['Stock_Recipient']-df_c['Stock_Recipient']), 2): >13}{                        round(float(df_t['Stock_Nonrecipient'] - df_c['Stock_Nonrecipient']),2): >13}{                          round(float(df_c['Stock']),2) :>13}")
print(f"FM margin      {round(float(df_t['Profit_Recipient']/df_t['Revenue_Recipient']- df_c['Profit_Recipient']/df_c['Revenue_Recipient']) , 2): >13}{round(float(df_t['Profit_Nonrecipient']/df_t['Revenue_Nonrecipient']- df_c['Profit_Nonrecipient']/df_c['Revenue_Nonrecipient']) ,2): >13}{round(float(df_c['Profit']/df_c['Revenue']),2) :>13}")


print(f"Expenditure Firm: {conversion * np.mean([hh.demand for hh in model_t.all_agents if hh.firm != None])}")
print(f"Expenditure No Firm: {conversion * np.mean([hh.demand for hh in model_t.all_agents if hh.firm == None])}")

print(f"Unemployment T: {df_t['Unemployment']}")
print(f"Unemployment C: {df_c['Unemployment']}")



#%%
"""
Calculate other intervention outcomes:
- Inflation 
- Unemployment
- Gini
- Multiplier 
"""


price_c = np.mean([fm.price for fm in model_c.all_firms])
price_t = np.mean([fm.price for fm in model_t.all_firms])


print(F"Change Inflation {price_t- price_c}")

#define function to calculate Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

# Calculate the Gini coefficient
incomes_c = df_hh_c[df_hh_c['Village'] == 1]['Expenditure']
incomes_t = df_hh_t[df_hh_t['Village'] == 1]['Expenditure']
Gini_t_vt = gini(incomes_t)
Gini_c_vt = gini(incomes_c)

print(f'Change Gini: Treated Villages, {Gini_t_vt-Gini_c_vt}')
print(f'Change Gini: Treated Villages, {Gini_t_vt-Gini_c_vt}')



def multiplier(GDP_t, GDP_c):
  dif_GDP = (GDP_t - GDP_c) * len(model_c.all_agents)
  transfer_size = len(model_t.intervention_handler.treated_agents)*1000
  multiplier = (1/transfer_size) * dif_GDP
  return multiplier

income_t = df_sm_t[df_sm_t['step'] == 142]['Expenditure']
income_c = df_sm_c[df_sm_c['step'] == 142]['Expenditure']
money_t = df_sm_t[df_sm_t['step'] == 142]['Money']
money_c = df_sm_c[df_sm_c['step'] == 142]['Money']

GDP_t = income_c + money_t
GDP_c = income_c + money_t

multiplier = multiplier(GDP_t, GDP_c)

print(f"Multiplier: {multiplier}")


#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compare_density(data_o, data_c):# Generate sample data for demonstration
  data1 = np.random.normal(0, 1, size=1000)
  data2 = np.random.normal(5, 2, size=1000)

  # Plot the first distribution using seaborn for density plot
  sns.kdeplot(data_o, bw_adjust=0.5, fill=True, alpha=0.5, label='Data 1')

  # Plot the second distribution on the same axis
  sns.kdeplot(data_c, bw_adjust=0.5, fill=True, alpha=0.5, label='Data 2')

  # Optionally adjust the y-axis limit
  # plt.ylim(0, <appropriate max value>)
  plt.xlim(0, 50)
  plt.legend()
  plt.show()


data_hh_o = read_dataframe("GE_Enterprise_ECMA.dta", "df")
data_fm_o = read_dataframe("GE_ENT-Survey-BL_Analysis_ECMA.dta", "df")

_, data_hh_c, data_fm_c  = model_c.datacollector.get_data()


data_hh_o.replace(0.00, np.nan, inplace=True)  
data_hh_o.dropna(subset=['ent_profit2_wins_PPP_BL', 'ent_revenue2_wins_PPP_BL'], inplace=True)  

profit_o = data_hh_o['ent_profit2_wins_PPP_vBL']
revenue_o = data_hh_o['ent_revenue2_wins_PPP_BL']
print(np.mean(revenue_o))
print(np.mean(profit_o))

#compare_density(revenue_o, revenue_c)
plot_dist(revenue_o, 'Revenue', (0,20))
plot_dist(profit_o, 'Profit', (0,20))

#%%



















































































#%%
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

print(f" Treated firm: {len([agent for agent in model_t.treated_agents if agent.firm != None])}")
print(f" Treaded no firm: {len([agent for agent in model_t.treated_agents if agent.firm == None])}")
print(f" Treaded empleyd: {len([agent for agent in model_t.treated_agents if agent.employer != None])}")
print(f" Treaded unemployed: {len([agent for agent in model_t.treated_agents if  agent.employer == None])}")
print(f" Treaded productivity: {np.mean([agent.productivity for agent in model_t.treated_agents])}")
print(f" Total average productivity: {np.mean([agent.productivity for agent in model_t.all_agents])}")

hh_df_recipient_t = df_hh_t[df_hh_t['Treated'] == 1]
hh_df_non_recipient_t = df_hh_t[df_hh_t['Treated'] == 0]                              
fm_df_recipient_t = df_fm_t[df_fm_t['Treated'] == 1]
fm_df_non_recipient_t = df_fm_t[df_fm_t['Treated'] == 0]

hh_df_recipient_c = df_hh_c[df_hh_c['Treated'] == 1]
hh_df_non_recipient_c = df_hh_c[df_hh_c['Treated'] == 0]                               
fm_df_recipient_c = df_fm_c[df_fm_c['Treated'] == 1]
fm_df_non_recipient_c = df_fm_c[df_fm_c['Treated'] == 0]

data = df_sm_c[df_sm_c['step'] == 52] 

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
model_t = Model()
model_t.datacollector = Sparse_collector(model_t)
model_t.run_simulation(360)

# get simulated data
df_sm, df_hh, df_fm, df_md, df_td = model_t.datacollector.get_calibration_data()

print(np.mean(df_hh['demand']))
print(np.mean(df_hh['money']))


data_hh_o = read_dataframe("GE_HHLevel_ECMA.dta", "df")['p2_consumption_PPP']/52
data_c = df_hh['demand']
compare_dist(data_hh_o, data_c, 'Expenditure Density', (0,100))