#%%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import qmc
import ABM
from data_collector import Sparse_collector
import timeit
import arviz as az
import dill
from scipy.stats import multivariate_normal

# Parameters to be calibrated
#====================================
# theta: probability for price change
# nu   : rate for price change
# phi_l: lower inventory rate 
# phi_u: uper inventory rate 
#
# alpha: cobb douglas demand parameter
# shape: parameter for gamma distribution (employee productivity)
# scale: parameter for gamme distribution (employee productivity)
data = {'Type': ['fm', 'fm', 'fm', 'fm', 'hh', 'hh', 'hh'], 
        'Name': ['theta', 'nu', 'phi_l', 'phi_u', 'alpha', 'mu', 'sigma'],
        'Bounds': [(0,1), (0.05, 0.5), (0.05, 0.5), (0.5, 1.5), (0,1), (0,2), (2,4)]} 
  
# Create DataFrame 
df_para = pd.DataFrame(data) 

true_thetas = torch.tensor([0.8, 0.3, 0.1, 1, 0.8, 1, 2], dtype=torch.float32)

class Model(ABM.Sugarscepe):
  """
  Type:        Child Class Sugarscepe
  Description: Implements the following additional functionality imperative for the calibration process:
               -set_parameters(): sets the parameters of the model to the specified values
               -run_simulaiton(): runs the simulation to generate the output vector x
               -sparse_datacollector: collects only the data necessary for the calibraiton process
  """
  def __init__(self):
    super().__init__()
    self.datacollector = Sparse_collector(self)
    
  def set_parameters(self, samples):
    """
    Type:        Method
    Description: Overwites the model parameters as specified in the 'parameters' argument
    """
    # create 
    fm_parameter_names = df_para[df_para['Type'] == 'fm']['Name'].tolist()
    hh_parameter_names = df_para[df_para['Type'] == 'hh']['Name'].tolist()

    fm_parameters =  { k:v for k, v in samples.items() if k in fm_parameter_names}
    hh_parameters =  { k:v for k, v in samples.items() if k in hh_parameter_names}

    for firm in self.all_firms:
      for parameter, value in fm_parameters.items():
        setattr(firm, parameter, value)
    
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
  Description: Crates the training data for the DNN 
  """
  # create a deep copy of the initial model state so model __init__ is only executed once
  simulation = Model()
  with open('../data/model_data/simulation_initial_state.dill', 'wb') as file:
    dill.dump(simulation, file)

  # create 2**model_runs theta values
  draws = create_sample_parameters(df_para, model_runs) 
  draws = np.concatenate([draws] * model_r, axis=0)  
  
  # create a list of dictionaries; each dictionary corresponds to one sample
  parameter_names = df_para['Name'].tolist()
  samples = [dict(zip(parameter_names, draw)) for draw in draws] 
  
  # run the model for each parameter sample generated
  # @Question run it for each sample more than once?
  x = []
  for i, sample in enumerate(samples):
    start = timeit.default_timer()
    # load a copy of the initial state of the simulation 
    with open('../data/model_data/simulation_initial_state.dill', 'rb') as file:
      sim_copy = dill.load(file)

    # Set the parameter values in the simulation copy
    sim_copy.set_parameters(sample)

    # run the simulation with the specified parameter values
    sim_copy.run_simulation(model_steps)

    # get the x vector i.e. the desired summary statistics
    ABM_output, _, _ = sim_copy.datacollector.get_calibration_data()
    x.append(ABM_output)
    end = timeit.default_timer()
    print(f"\rSimulating step: {i + 1} ({round((i + 1)*100/len(samples), 0)}% complete) time: {round(end-start, 3)}", end="", flush=True)

  print("\n")

  # Create a troch tensor from the sample parameter values, theta, and the output vectors, x
  theta = torch.Tensor(draws) 
  x = torch.Tensor(x)

  # Create a torch data set for model training 
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
  
  # if load == False a new dataset is created in 'create_dataset()'
  else:
    dataset = create_dataset(model_runs, model_steps, model_r)

  # Split the dataset into training and testing sets
  total_samples = len(dataset)
  train_size = int(0.8 * total_samples)
  test_size = total_samples - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  
  # Create DataLoader instances for training and testing
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return train_dataloader, test_dataloader

# define hyper-parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 1000
I = df_para.shape[0]  # input size
O =  11 # output size = nr of summary statistics
eta = 0.001 # learning rate

def hyperparameter_tuning():
  """
  Type:        Helper Function 
  Description: For surrogate model selection
  References:  https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
  """
  # @TODO try:
  # Batch normalization layer: how? doesnt agree with rejection abc
  # Adaptive laerning rate
  # Node size
  # transform in dataloader?
  # momentum?
  pass

# Define the network class
class Surrogate(nn.Module):
    def __init__(self):
        super(Surrogate, self).__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(I, 100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, O),
        )

    def forward(self, x):
        Z = self.linear_sigmoid_stack(x)
        return Z

def plot_losses(train_losses, test_losses):
  """
  Type:        Helper function 
  Description: Plots training and test losses
  """
  # @make this pretty
  #@ plot training predicitons vs true model using test data
  plt.plot(train_losses, label='train_loss')
  plt.plot(test_losses, label='val_loss')
  plt.show()     

def train(train_loader, test_loader, model, loss_fn, optimizer, epochs):
  """
  Type:        Function 
  Description: Trains the surrogate model 
  """
  train_losses = []
  
  # train the DNN
  for epoch in range(epochs):
      print(f'\repoch: {epoch}', end='', flush=True)
      running_loss = 0.0
      for theta, x in train_loader:

        # reset the gradient
        optimizer.zero_grad()

        # Compute prediction and its error
        pred = model(theta)
        loss = loss_fn(pred, x)
        loss.backward()

        # Perform weight update
        optimizer.step()

        # add loss of current batch
        running_loss += loss.item()

      # collect the average test loss for the current epoch
      train_losses.append(running_loss / len(train_loader))

  # test the DNN 
  with torch.no_grad():
    running_loss = 0
    test_losses = []

    # Use designated test data to assess the predictions
    for theta, x in test_loader:

      # Compute network output
      pred = model(theta)

      # compute loss and add it to test_losses
      loss = loss_fn(pred, x)
      running_loss += loss.item()
      test_losses.append(running_loss / len(test_loader))

  # plot the losses
  plot_losses(train_losses, test_losses)

# get train and test data as dataloaders
train_loader, test_loader = create_dataloader(model_runs=7, model_steps=100, batch_size=24, model_r=5, load=True)

# Instantiate network with MSE loss and Adam optimizer
surrogate = Surrogate().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=surrogate.parameters(), lr=eta)

# Train the surrogate model
train(train_loader, test_loader, surrogate, loss_fn, optimizer, epochs)

def calculate_L2(x, y):
   """
   Type:        Helper function 
   Description: Calculates the L2 vector norm
   """
   return torch.norm(x - y, 2).item()

def rejection_abc(y, surrogate):
  """
  Type:        Function 
  Description: Main function to conduct Baysian inference 
  """
  # create 2^17 = 131 072 theta values
  draws = create_sample_parameters(df_para, m=17)    
  
  # for each theta, predict the x vectors using the surrogate and calculate the corresponding mse
  samples = []
  surrogate.eval()

  for draw in draws:
    with torch.no_grad():
      draw = torch.tensor(draw, dtype=torch.float32) #.unsqueeze(0)
      x_pred = surrogate(draw)
      mse = calculate_L2(x_pred, y)
    samples.append((draw, x_pred, mse))

  # order the list in descending order and keep the 0.1% of the x-predictions most close to the actually observed outcome
  samples = sorted(samples, key=lambda sample : sample[2])
  selected_samples = samples[:int(len(samples) * 0.001)]
  best_guess = min(selected_samples, key=lambda x: x[2])

  # get theta values and convert them into an np.array
  selected_thetas = [sample[0].cpu().data.numpy() for sample in selected_samples]
  #return selected_thetas, best_guess

  observations = np.vstack(selected_thetas)

  # Initialize the proposal distribution https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf
  cov_matrix = np.cov(observations, rowvar=False)*0.01#0.1 * np.eye(df_para.shape[0]) 
  mean_values = np.mean(observations, axis=0)
  proposal_dist = multivariate_normal(mean=mean_values, cov=cov_matrix)

  # Set the threshhold to the best guess
  epsilon = best_guess[2]#selected_samples[int(len(selected_samples) * 0.01)][2]

  # plot the posterior density
  ABC_MCMC = []
  i = 0
  mse_min = best_guess[2]
  while len(ABC_MCMC) < 250:
    with torch.no_grad():
      proposed_theta = torch.Tensor(proposal_dist.rvs(size=1)) # .unsqueeze(0)
      x_pred = surrogate(proposed_theta)
      mse = calculate_L2(x_pred, y)

      if mse < epsilon:
        ABC_MCMC.append(proposed_theta)
        proposal_dist = multivariate_normal(mean=proposed_theta, cov=cov_matrix)

        if mse < mse_min:
          best_guess = (proposed_theta, x_pred, mse)
      i+= 1
      print(f'\r Trials{i}. Found {len(ABC_MCMC)}' , flush=True, end='')

  return ABC_MCMC, best_guess


def plot_distributions(estimated_thetas, true_thetas):
  fig, ax = plt.subplots(ncols=1, nrows=7, constrained_layout=True)

  # create a distribution plot for each theta
  for i, parameter in enumerate(df_para['Name'].values):
    az.plot_dist( [theta[i] for theta in estimated_thetas], ax=ax[i], label="Surrogate Output", rug = True, color='#0E7580', quantiles=[.05, .5, .95] )
    # Add true value as line, labels, title and legend 
    ax[i].axvline(x=true_thetas[i], color='#800E0E', linestyle='--', label='Vertical Line at 0')
    ax[i].set_xlabel("Value")
    ax[i].set_ylabel("Density")
    ax[i].set_title(f"Kernel Density {parameter}")
    ax[i].legend()

  # Show the plot
  plt.show()

m = Model()
m.run_simulation(100)
y, _, _ = m.datacollector.get_calibration_data()
y = torch.tensor(y, dtype=torch.float32)

estimated_thetas, best_guess = rejection_abc(y, surrogate)
print(f'true theta: {true_thetas}')
print(f'estimated theta: {best_guess[0]}')

plot_distributions(estimated_thetas, true_thetas)
# %%
