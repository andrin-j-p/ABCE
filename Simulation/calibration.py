#%%
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import qmc, wasserstein_distance
import ABM
from datacollector import Sparse_collector
import timeit
import arviz as az
import dill
import random
from read_data import read_dataframe
import seaborn as sns

np.random.seed(0) 
random.seed(0)

# Parameters to be calibrated
#====================================
# lambda:firm productivity
# theta: probability for price change
# nu   : rate for price change
# phi_l: lower inventory rate 
# phi_u: uper inventory rate 
# alpha: cobb douglas demand parameter
# shape: parameter for gamma distribution (employee productivity)
# scale: parameter for gamme distribution (employee productivity)

data = {'Type': ['fm', 'fm', 'fm', 'fm', 'fm', 'hh', 'hh', 'hh'], 
        'Name': [ 'productivity',  'theta',   'nu',      'phi_l',    'hi_u', 'alpha',    'mu',   'sigma'],
        'Bounds': [(0.5, 1.5),    (0.5,1), (0.05, 0.5),  (0, 0.5), (0.5, 1), (0,0.9), (3, 4), (0.2, 1)]} 
  
# Convert the paramters into a pandas DataFrame 
df_para = pd.DataFrame(data) 

class Model(ABM.Model):
  """
  Type:        Child Class Sugarscepe
  Description: Implements the following additional functionality imperative for the calibration process:
               -set_parameters(): sets the parameters of the model to the specified values
               -run_simulaiton(): runs the simulation to generate the output vector x
               -sparse_datacollector(): collects only the data necessary for the calibraiton process
  """
  def __init__(self):
    super().__init__()
    self.datacollector = Sparse_collector(self)
    
  def set_parameters(self, samples):
    """
    Type:        Method
    Description: Overwites the model parameters as specified in the 'parameters' argument
    """

    # create list of firm and agent paramters names
    fm_parameter_names = df_para[df_para['Type'] == 'fm']['Name'].tolist()
    hh_parameter_names = df_para[df_para['Type'] == 'hh']['Name'].tolist()

    # create dictionary with paramter name as key and paramter value as value
    fm_parameters =  { k:v for k, v in samples.items() if k in fm_parameter_names}
    hh_parameters =  { k:v for k, v in samples.items() if k in hh_parameter_names}

    # set firm paramters
    for firm in self.all_firms:
      for parameter, value in fm_parameters.items():
        setattr(firm, parameter, value)
    
    # set agent paramters
    for hh in self.all_agents:
      for parameter, value in hh_parameters.items():
        setattr(hh, parameter, value)

  def run_simulation(self, steps):
    """
    Type:        Method
    Description: Overwrite run simulaiton in parent class to avoid print statement
    """
    for _ in range(steps):
      self.step()
    

def create_sample_parameters(parameters, m):
    """
    Type:           Helper function
    Description:    Creates N sample parameter vectors to train the DNN
                    The samples are pseudo-random and based on the Sobol sequence to speed up convergence
    References:     https://iopscience.iop.org/article/10.3847/0004-637X/830/1/31/meta
                    Jorgenson 2022
                    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
    """
    # create a Sobol sequence with dimension d='number of parameters' and length 2^m
    sampler = qmc.Sobol(d=parameters.shape[0], scramble=False)
    draws = sampler.random_base2(m=m)

    # scale the sequence to the parameter range
    l_bounds = [l[0] for l in parameters.Bounds]
    u_bounds = [u[1] for u in parameters.Bounds]
    draws = qmc.scale(draws, l_bounds, u_bounds)
    draws = np.round(draws, 5)

    return draws


def create_dataset(model_runs, model_steps, model_r):
  """
  Type:        Function
  Description: Crates the dataset for rejection ABC  
  """
  # Store a deep copy of the initial model state so model __init__ is only executed once

  # create 2**model_runs theta values
  draws = create_sample_parameters(df_para, model_runs) 
  draws = np.concatenate([draws] * model_r, axis=0)  
  
  # create a list of dictionaries; each dictionary corresponds to one sample
  parameter_names = df_para['Name'].tolist()
  samples = [dict(zip(parameter_names, draw)) for draw in draws] 
  
  # run the model for each parameter sample generated
  x = []
  for i, sample in enumerate(samples):
    sim = Model()

    start = timeit.default_timer()

    # Set the parameter values in the simulation copy
    sim.set_parameters(sample)

    # run the simulation with the specified parameter values
    sim.run_simulation(model_steps)

    # get the x vector the expenditure vector summary statistics
    ABM_output = sim.datacollector.get_calibration_data()
    x.append(ABM_output)

    # print status 
    end = timeit.default_timer()
    print(f"\rSimulating step: {i + 1} ({round((i + 1)*100/len(samples), 0)}% complete) time: {round(end-start, 3)}", end="", flush=True)
  
  print("\n")

  # Create a troch tensor from the sample parameter values, theta, and the output vectors, x
  theta = torch.Tensor(draws) 
  x = torch.Tensor(x)

  # Convert data into Tensordataset to create the Dataloader object 
  dataset = TensorDataset(theta, x) 

  # Save the dataset as a file
  with open('../data/model_data/data_loader1.dill', 'wb') as file:
    dill.dump(dataset, file)

  return dataset


def create_dataloader(model_runs, model_steps, batch_size, load, model_r):
  """
  Type:        Function
  Description: Creates a pytorch dataloader from a dataset 
               used in the training loop
  """
  # if load == true the previously saved dataset is loaded
  # note: the size of the dataset will not necessarily match the size specified in m
  if load == True:
    # To load the dataset from the saved file
    with open('../data/model_data/data_loader1.dill', 'rb') as file:
        loaded_dataset = dill.load(file)
    
    dataset = loaded_dataset
  
  # If load == False a new dataset is created in 'create_dataset()'
  else:
    dataset = create_dataset(model_runs, model_steps, model_r)
  
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return data_loader


def rejection_abc(data_loader, y):
  """
  Type:        Function 
  Description: Main function for Baysian inference 
  """
  # for each theta, predict the x vectors using the surrogate and calculate the corresponding mse
  best_guess = (0, 0, 10000)
  total_sample = []
  for theta, x in data_loader:
    x = np.squeeze(x.numpy())
    theta = np.squeeze(theta.numpy())
    wd = wasserstein_distance(list(x), list(y))

    sample = (theta, x, wd)

    if wd < best_guess[2]:
      best_guess = sample

    total_sample.append(sample)

  # Sort all samples based on the wasserstein distance loss and retain the best 10%
  final_sample = sorted(total_sample, key=lambda sample : sample[2])[:12]

  return best_guess, final_sample


def compare_dist(data_true, data_sim):
  """
  Type:        Function 
  Description: Compare two distributions  
  """
  # set arviz display style
  az.style.use("arviz-doc")

  # Crate plot
  fig, ax = plt.subplots()
  az.plot_dist(data_true, ax=ax, label="true", rug = True, rug_kwargs={'space':0.1}, fill_kwargs={'alpha': 0.7})
  az.plot_dist(data_sim, ax=ax, label="pred", color='red', rug=True,  rug_kwargs={'space':0.2}, fill_kwargs={'alpha': 0.7})
  
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density Comparision True and Simulated Expenditures")
  ax.legend()

  # Show the plot
  plt.xlim(0, 200)
  plt.show()

# Construct dataloader object
data_loader = create_dataloader(model_runs=7, model_steps=1000, batch_size=1, model_r=1, load=True)

# Get ovserved expenditure data for validation 
y = read_dataframe("GE_HHLevel_ECMA.dta", "df")
y = y[(y['hi_sat']==0) & (y['treat'] == 0)]['p2_consumption_PPP'].dropna().values/52

# Conduct Bayesian inference and get best guess of the parameter values
best_guess, final_sample = rejection_abc(data_loader, y)
x = best_guess[1]

compare_dist(y, x)
print(best_guess[0])

def create_boxplots(data_tuples, limits, titles):
    # Unzip the tuples into separate lists
    data_lists = list(zip(*data_tuples))
    
    # Set up the figure
    fig, axes = plt.subplots(2, 4, figsize=(30, 15))  # Creates a 2x5 grid of subplots
    
    # Flatten the 2D axes array to a 1D array for easier indexing
    axes_flat = axes.flatten()
    
    # Loop through each list (corresponding to tuple elements) to create a boxplot
    for i, data in enumerate(data_lists):
        # Use the flattened axes array for indexing
        ax = axes_flat[i]
        ax.boxplot(data, medianprops=dict(color='#2C72A6', linewidth=2))
        ax.set_title(titles[i], pad=45, fontsize=22)
        if i % 4 == 0:
            ax.set_ylabel('Parameter Value', fontsize=20, labelpad=20) 
        ax.set_ylim(limits[i])
        ax.set_xticklabels([])


    plt.tight_layout()  # Adjust the rect if legend overlaps with subplots
    fig.subplots_adjust(hspace=0.5) 
    
    plt.savefig('../data/illustrations/boxplot.png', dpi=700, bbox_inches='tight')

    plt.show()

paramter_values = [p[0] for p in final_sample]
limits = data['Bounds']
titles = [r'Firm Productivity $\lambda$', r'Probability of Price Change $\theta$', r'Rate of Price Change $\nu$', r'Lower Inventory Rate $\phi_l$', 
          r'Upper Inventory Rate $\phi_u$', r'Propensity to Consume $\alpha$', r'Household Productivity $\mu$', r'Households Productivity $\sigma$']

create_boxplots(paramter_values, limits, titles)



# %%
