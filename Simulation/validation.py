#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ABM import Model 
from read_data import read_dataframe
from datacollector import Validation_collector
from intervention_handler import Intervention_handler2
import arviz as az
import matplotlib.patches as mpatches
import seaborn as sns
import random

np.random.seed(0) 
random.seed(0)

steps = 360
model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
data_md_c = model_c.datacollector.get_data()
data_fm_c = model_c.datacollector.fm_df
data_hh_c = model_c.datacollector.hh_df
#%%
"""
State of the economy before intervention:
- Expenditure
- Income
- Assets
- Profit
- Revenue

- Firm size
"""    

def compare_histograms(data_o, data_c):
  bins = np.arange(1, 17)
  density_o, _ = np.histogram(data_o, bins=bins, density=True)
  density_c, _ = np.histogram(data_c, bins=bins, density=True)

  fig, ax = plt.subplots()  
  pos = np.arange(15)
  wid = 0.3
  plt.bar(pos,  density_c, width=wid, color='#579eb1' , label='Simulated')
  plt.bar(pos + wid, density_o, width=wid, color='#52C290', label='Observed')

  # Add labels, title, legend, etc.
  ax.set_ylabel("")
  ax.set_xlabel("Value")
  ax.set_title(f"Number of Firm Employees")
  ax.legend()
  plt.xticks(pos + wid / 2, [i for i in range(1,16)])
  plt.show()


def calibration_dist(data_true, data_sim):
  """
  Type:        Function 
  Description: Compare two distributions  
  """
  # set arviz display style
  az.style.use("arviz-doc")

  # Crate plot
  fig, ax = plt.subplots()
  az.plot_dist(data_sim, ax=ax, label="Simulated", color='#579eb1', rug=True,  rug_kwargs={'color': '#579eb1', 'space':0.2}, fill_kwargs={'alpha': 1})
  az.plot_dist(data_true, ax=ax, label="Observed", color='#52C290', rug = True, rug_kwargs={'color': '#52C290', 'space':0.1}, fill_kwargs={'alpha': 0.7})
  
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Household Expenditures")
  ax.set_title(f"Observed vs. Simulated Expenditures")
  ax.legend()

  # Show the plot
  plt.xlim(0, 200)
  
  plt.savefig('calibration.png', dpi=700, bbox_inches='tight')    

  plt.show()


def plot_dist(data, title, limit, ax=None, color='blue', xlabel='', ylabel=''):
    if ax is None:
        fig, ax = plt.subplots()
    
    az.plot_dist(data, ax=ax, rug=False, color=color, fill_kwargs={'alpha': 1})
    ax.set_ylabel(f"{ylabel}")
    ax.set_xlabel(f"{xlabel}")
    ax.set_title(f"{title}")
    ax.set_xlim(limit[0], limit[1])


def compare_dist(datasets, titles, limits):
    fig, axes = plt.subplots(2, 5, figsize=(30, 10))  # Correctly note it's a 2x5 grid
    axes = axes.flatten()
    
    legend_handles = []
    legend_labels = []

    for i, (ax, data, title, limit) in enumerate(zip(axes, datasets, titles, limits)):
        row = i // 5  # Determines the row based on index
        color = '#579eb1' if row == 0 else '#52C290'
        ylabel = '' if i % 5 != 0 else 'Density'  # Adjust for a 2x5 grid
        xlabel = title
        modified_title = title + ' Simulated' if row == 0 else title + ' Observed'
        
        # Assuming plot_dist is a function you've defined elsewhere
        plot_dist(data, modified_title, limit, ax=ax, color=color, xlabel=xlabel, ylabel=ylabel)
        
        # Set title with larger font size
        ax.set_title(modified_title, fontsize=16)  # Increase fontsize as needed
        ax.set_xlabel(xlabel, fontsize=14)  # Increase fontsize for x-axis title
        ax.set_ylabel(ylabel, fontsize=14) 
        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    legend_handles.extend([mpatches.Patch(color='#579eb1'), mpatches.Patch(color='#52C290')])
    legend_labels.extend(['Simulated', 'Observed'])

    # Increase space between the first and second row
    fig.subplots_adjust(hspace=0.5)  

    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, frameon=False, fontsize='x-large')

    plt.savefig('pre_intervention.png', dpi=700, bbox_inches='tight')    

    plt.show()



def clean_data(data):
  data.replace(0.00, np.nan, inplace=True)  
  data.dropna(inplace=True)  
  return data


data_hh_o = read_dataframe("GE_HH-Survey-BL_Analysis_AllHHs.dta", "df")
data_fm_o = read_dataframe("GE_ENT-Survey-BL_Analysis_ECMA.dta", "df")

#Expenditure
expenditure_o = read_dataframe("GE_HHLevel_ECMA.dta", "df")['p2_consumption_PPP']/52
expenditure_c = data_hh_c['Expenditure']

#Assets 
money_o = data_hh_o['p1_assets_trim_PPP_BL']*0.1
money_c = data_hh_c[data_hh_c['Assets']>1]['Assets']

#Income
income_o = data_hh_o['emp_income']/520*1.781
income_o = clean_data(income_o)
income_c = data_hh_c[data_hh_c['Income']>=1]['Income']

# Revenue
revenue_o = read_dataframe("GE_Enterprise_ECMA.dta", "df")
revenue_o = revenue_o[(revenue_o['treat'] == 0) & (revenue_o['hi_sat'] == 0)]
revenue_o = revenue_o['ent_revenue1_wins_PPP']/52
revenue_o = clean_data(revenue_o)
revenue_c = data_fm_c[data_fm_c['Revenue']>1]['Revenue']

#Profit
profit_o = read_dataframe("GE_Enterprise_ECMA.dta", "df")
profit_o = profit_o[(profit_o['treat'] == 0) & (profit_o['hi_sat'] == 0)]
profit_o = profit_o['ent_profit2_wins_PPP']/52
profit_c = data_fm_c[data_fm_c['Profit']>1]['Profit']


datasets = [expenditure_c, money_c, income_c, revenue_c, profit_c, expenditure_o, money_o, income_o, revenue_o, profit_o] 

titles = ["Expenditure", "Assets", "Income", "Revenue", "Profit", "Expenditure", "Assets", "Income", "Revenue", "Profit"]

limits = [(1, 200), (1, 300), (1, 150), (1,500), (1,300), (0, 200), (1, 300), (1, 150), (1,30), (0,20)]  

compare_dist(datasets, titles, limits)

#%%
# Get calibration outcome
y = read_dataframe("GE_HHLevel_ECMA.dta", "df")
expenditure_o = y[(y['hi_sat']==0) & (y['treat'] == 0)]['p2_consumption_PPP'].dropna().values/52
expenditure_c = data_hh_c[data_hh_c['Expenditure'] > 1]['Expenditure']

calibration_dist(expenditure_o, expenditure_c)

#%%
# Number Employees
df = read_dataframe("GE_Enterprise_ECMA.dta", "df")
df.dropna(subset=['emp_n_tot'], inplace=True)
df = df[df['emp_n_tot'].apply(lambda x: 0 < x <= 15)]
employees_o = df['emp_n_tot']

df_sm_c_EL = data_fm_c[data_fm_c['Employees'].apply(lambda x: 0 < x <= 15)]
employees_c = df_sm_c_EL['Employees']

compare_histograms(employees_o, employees_c)

# Market Clearing
print(f"Demand Satisfied: {data_md_c[data_md_c['step'] == 51]['Demand_Satisfied'].values[0]}")


#%%
#DELETE
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ABM import Model 
from read_data import read_dataframe
from datacollector import Validation_collector
import arviz as az
import pandas as pd
import seaborn as sns
import random

steps = 1000

np.random.seed(0) 
random.seed(0)

model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
df_hh_c = model_c.datacollector.hh_df
df_sm_c = model_c.datacollector.get_data()

np.random.seed(0) 
random.seed(0)

model_t = Model()
model_t.datacollector = Validation_collector(model_t)
model_t.intervention_handler.control = False
model_t.run_simulation(steps)
df_hh_t = model_t.datacollector.hh_df
df_sm_t = model_t.datacollector.get_data()

#%%
"""
Calculates main intervention outcomes:
- Overall Statistics (Number HH, FM etc.)
- Expenditure
- Money
- Income
- Revenue
- Profit
- Price 
- Unemployment
"""

def create_lineplots(df_t, df_c):
    variables = ['Expenditure', 'Assets', 'Income', 'Revenue', 'Profit', 'Unemployment']

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
        ax.set_xlabel('Week', fontsize=14)
        ax.set_ylabel(var, fontsize=14)
        ax.set_title(f"{var}", fontsize=16)

        # If it's the first iteration, capture the legend handles and labels
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Create a custom legend handle for the vertical line
            red_line_handle = Line2D([0], [0], color='red', linestyle='--', label='Time of UCT')
            # Add them to the list of handles and labels
            handles.extend([red_line_handle])           
            labels.extend(['Time of UCT'])
            legend_handles.extend(handles)
            legend_labels.extend(labels)

    # Create a single legend for the entire figure using the handles and labels captured
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize='x-large')

    # Adjust layout for better spacing

    plt.tight_layout()  # Adjust the rect if legend overlaps with subplots
    fig.subplots_adjust(hspace=0.5) 

    # Show plot
    plt.savefig('post_intervention.png', dpi=700, bbox_inches='tight')
    plt.show()


create_lineplots(df_sm_t, df_sm_c)


# From here on, only observation 18 months after intervention are used
df_sm_t_EL = df_sm_t[df_sm_t['step'] == 142] 
df_sm_c_EL = df_sm_c[df_sm_c['step'] == 142] 

conversion = 52
print(f"               {'Recipients': >13}{'Nonrecipinets':>13}{'Control':>13}")
print(f"HH Expenditure {round(float(df_sm_t_EL['Expenditure_Recipient']-df_sm_c_EL['Expenditure_Recipient'])*conversion , 2): >13}{round(float(df_sm_t_EL['Expenditure_Nonrecipient'] - df_sm_c_EL['Expenditure_Nonrecipient'])*conversion , 2) : >13}{round(float(df_sm_c_EL['Expenditure'])*conversion ,2) :>13}")
print(f"HH Assets      {round(float(df_sm_t_EL['Assets_Recipient']-df_sm_c_EL['Assets_Recipient']), 2): >13}{                        round(float(df_sm_t_EL['Assets_Nonrecipient'] - df_sm_c_EL['Assets_Nonrecipient']), 2 ) : >13}{                       round(float(df_sm_c_EL['Assets']),2):>13}")
print(f"HH Income      {round(float(df_sm_t_EL['Income_Recipient']-df_sm_c_EL['Income_Recipient'])*conversion , 2) : >13}{         round(float(df_sm_t_EL['Income_Nonrecipient'] - df_sm_c_EL['Income_Nonrecipient'] )*conversion ,2): >13}{           round(float(df_sm_c_EL['Income'])*conversion ,2) :>13}")
print(f"FM Profit      {round(float(df_sm_t_EL['Profit_Recipient']-df_sm_c_EL['Profit_Recipient'])*conversion , 2) : >13}{         round(float(df_sm_t_EL['Profit_Nonrecipient'] - df_sm_c_EL['Profit_Nonrecipient'])*conversion ,2) : >13}{           round(float(df_sm_c_EL['Profit'])*conversion ,2) :>13}")
print(f"FM Revenue     {round(float(df_sm_t_EL['Revenue_Recipient']-df_sm_c_EL['Revenue_Recipient'])*conversion , 2): >13}{        round(float(df_sm_t_EL['Revenue_Nonrecipient'] - df_sm_c_EL['Revenue_Nonrecipient'])*conversion ,2): >13}{          round(float(df_sm_c_EL['Revenue'])*conversion ,2) :>13}")
print(f"FM Margin      {round(float(df_sm_t_EL['Profit_Recipient']/df_sm_t_EL['Revenue_Recipient']- df_sm_c_EL['Profit_Recipient']/df_sm_c_EL['Revenue_Recipient']) , 2): >13}{round(float(df_sm_t_EL['Profit_Nonrecipient']/df_sm_t_EL['Revenue_Nonrecipient']- df_sm_c_EL['Profit_Nonrecipient']/df_sm_c_EL['Revenue_Nonrecipient']) ,2): >13}{round(float(df_sm_c_EL['Profit']/df_sm_c_EL['Revenue']),2) :>13}")

#%%
# Statistics for treatment illustration 
low_saturation_hh = [hh for hh in model_t.all_agents if hh.village.market.saturation == 0]
high_saturation_hh = [hh for hh in model_t.all_agents if hh.village.market.saturation == 1]

low_saturation_vl = [vl for vl in model_t.all_villages if vl.market.saturation == 0]
high_saturation_vl = [vl for vl in model_t.all_villages if vl.market.saturation == 1]

print(f'Number HH: {len(model_t.all_agents)}')
print(f'Number FM: {len(model_t.all_firms)}')
print(f'Number VL: {len(model_t.all_villages)}')
print(f'Number MK: {len(model_t.all_markets)}')

print(f'Number HH LS-Market Total: {len(low_saturation_hh)}')
print(f'Number HH HS-Market Total: {len(high_saturation_hh)}')
print(f'Number HH HS-Market Treated: {len([hh for hh in high_saturation_hh if hh.treated == 1])}')
print(f'Number HH HS-Market Control: {len([hh for hh in high_saturation_hh if hh.treated == 0])}')
print(f'Number HH LS-Market Treated: {len([hh for hh in low_saturation_hh if hh.treated == 1])}')
print(f'Number HH LS-Market Control: {len([hh for hh in low_saturation_hh if hh.treated == 0])}')

print(f'Number VL LS-Market Total: {len(low_saturation_vl)}')
print(f'Number VL HS-Market Total: {len(high_saturation_vl)}')
print(f'Number VL HS-Market Treated: {len([vl for vl in high_saturation_vl if vl.treated == 1])}')
print(f'Number VL HS-Market Control: {len([vl for vl in high_saturation_vl if vl.treated == 0])}')
print(f'Number VL LS-Market Treated: {len([vl for vl in low_saturation_vl if vl.treated == 1])}')
print(f'Number VL LS-Market Control: {len([vl for vl in low_saturation_vl if vl.treated == 0])}')

 #%%
"""
Calculate other intervention outcomes:
- Inflation 
- Unemployment
- Gini
- Multiplier 
"""


### Unemployment

unemployment_t = float(df_sm_t_EL['Unemployment'].iloc[0])
unemployment_c = float(df_sm_c_EL['Unemployment'].iloc[0])
print(f"Unemployment T: {unemployment_t}")
print(f"Unemployment C: {unemployment_c}")
print(f"Change Unemployment Simulated: {(unemployment_c - unemployment_t) * 100}%")

### Inflation 

# Calculate change in inflation
price_c = df_sm_c_EL['Price'].mean()
price_t = df_sm_t_EL['Price'].mean()

print(F"Change Inflation {price_t - price_c}")


### Gini

# Function to calculate Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

# Calculate the Gini coefficient in treatment villages
incomes_c = df_hh_c[df_hh_c['Village'] == 1]['Expenditure']
incomes_t = df_hh_t[df_hh_t['Village'] == 1]['Expenditure']

# Calculate change in treatment village Gini Coefficient 
print(f'Change Treated Village GINI: {gini(incomes_t) - gini(incomes_c)}')


### Multiplier

def multiplier(GDP_t, GDP_c):
  dif_GDP = (GDP_t - GDP_c)* len(model_c.all_agents) 
  transfer_size = len(model_t.intervention_handler.treated_agents)*1000
  multiplier = (1/transfer_size) * dif_GDP
  return multiplier


expenditure_t = df_sm_t_EL['Expenditure'].values[0] * 116
expenditure_c = df_sm_c_EL['Expenditure'].values[0] * 116
stock_t = df_sm_t_EL['Stock'].values[0]
stock_c = df_sm_c_EL['Stock'].values[0]
print(stock_t)
print(stock_c)
GDP_t = expenditure_t + stock_t 
GDP_c = expenditure_c + stock_c

print(f"Multiplier: {multiplier(GDP_t, GDP_c)}")


#%%
model_t = Model()
model_t.datacollector = Validation_collector(model_t)
model_t.intervention_handler.control = False
model_t.run_simulation(150)
df_hh_t = model_t.datacollector.hh_df
df_sm_t = model_t.datacollector.get_data()
#%%

def multiplier2(delta_GDP):
  transfer_size = len(model_t.intervention_handler.treated_agents)*1000
  multiplier = (1/transfer_size) * delta_GDP
  return multiplier

# Consumption (all expenditures fall on consumption)
expenditure = np.array(df_sm_t['Expenditure'].tail(116))*52.2
expenditure = np.diff(expenditure)
expenditure = np.cumsum(expenditure)
expenditure = expenditure[-1] * len(model_t.all_agents)

# Investment (the only form of investment are changes in inventory)
stock = np.array(df_sm_t['Stock'].tail(116))
stock = np.diff(stock)
stock = np.cumsum(stock)[-1] * len(model_t.all_firms)

delta_GDP = expenditure + stock
print(multiplier2(delta_GDP))


### Spillovers
#%%
# Spillovers to eligible (i.e. poor) non-recipient households
print(f"Spillover to elligible: {(df_hh_t[(df_hh_t['Eligible'] == 1)  & (df_hh_t['Treated'] == 0)]['Income'].mean() - df_hh_c[(df_hh_c['Eligible'] == 1) & (df_hh_c['Treated'] == 0)]['Income'].mean()) * conversion}")
# Spiillovers to ineligible (i.e. rich) non-recipient households
print(f"Spillover to inelligible: {(df_hh_t[(df_hh_t['Eligible'] == 0)  & (df_hh_t['Treated'] == 0)]['Income'].mean() - df_hh_c[(df_hh_c['Eligible'] == 0)  & (df_hh_c['Treated'] == 0)]['Income'].mean()) * conversion}")

# Spillovers to ineligible households in control villages (across village spillover)
print(f"Intra-village Spillover: {(df_hh_t[(df_hh_t['Eligible'] == 0) & (df_hh_t['Village'] == 0)]['Income'].mean() - df_hh_c[(df_hh_c['Eligible'] == 0) & (df_hh_c['Village'] == 0)]['Income'].mean()) * conversion}")
# Spillovers to ineligible households in treatment villages (intra village spillover)
print(f"Inter-village Spillover: {(df_hh_t[(df_hh_t['Eligible'] == 0) & (df_hh_t['Village'] == 1)]['Income'].mean() - df_hh_c[(df_hh_c['Eligible'] == 0) & (df_hh_c['Village'] == 1)]['Income'].mean()) * conversion}")


### Long term impact
#%%

### Long term impact
model_t.run_simulation(2000)
df_sm_t = model_t.datacollector.get_data()

print(df_sm_t[df_sm_t['step'] ==  52]['Expenditure'])
print(df_sm_t[df_sm_t['step'] == 142]['Expenditure'])
print(df_sm_t[df_sm_t['step'] == 200]['Expenditure'])
print(df_sm_t[df_sm_t['step'] == 260]['Expenditure'] )
print(df_sm_t[df_sm_t['step'] == 400]['Expenditure'] )
#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ABM import Model 
from read_data import read_dataframe
from datacollector import Validation_collector
from intervention_handler import Intervention_handler2
import arviz as az
import pandas as pd
import seaborn as sns
import random


steps = 1000

np.random.seed(0) 
random.seed(0)

model_c = Model()
model_c.datacollector = Validation_collector(model_c)
model_c.intervention_handler = Intervention_handler2(model_c)
model_c.intervention_handler.control = True
model_c.run_simulation(steps)
df_hh_c = model_c.datacollector.hh_df
df_sm_c = model_c.datacollector.get_data()

np.random.seed(0) 
random.seed(0)

model_t = Model()
model_t.datacollector = Validation_collector(model_t)
model_t.intervention_handler = Intervention_handler2(model_t)
model_t.intervention_handler.control = False
model_t.run_simulation(steps)
df_hh_t = model_t.datacollector.hh_df
df_sm_t = model_t.datacollector.get_data()


# Function to calculate Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


# Calculate the Gini coefficient in treatment villages
incomes_c = df_hh_c[df_hh_c['Village'] == 1]['Expenditure']
incomes_t = df_hh_t[df_hh_t['Village'] == 1]['Expenditure']

# Calculate change in treatment village Gini Coefficient 
print(f'Change Treated Village GINI: {gini(incomes_t) - gini(incomes_c)}')

# Questions
# OK 1) How does intervention group look like? mostly firm owners? -> mostly employed workers with low productivity
# 2) Why dip in treated after token
# 3) Why demand/income/ money increases more for control
# 4) how does benefit differ firm owner and employees

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


# %%
