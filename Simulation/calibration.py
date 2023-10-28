import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import qmc
import ABM
import timeit
import dill

import cProfile # move to Test.py
import pstats   # move to Test.py

# Parameters to be calibrated
# theta: probability for price change
# nu:    rate for price change
# phi_l: lower inventory rate 
# phi_u: uper inventory rate 
data = {'Type': ['fm', 'fm', 'fm', 'fm'], 
        'Name': ['theta', 'nu', 'phi_l', 'phi_u'],
        'Bounds': [(0,1), (0,5), (0.1, 0.5), (0.5, 10)]} 
  
# Create DataFrame 
df_para = pd.DataFrame(data) 

def create_sample_parameters(parameters, m):
    """
    Type:           Helper function
    Description:    Creates N sample parameter vectors to train the DNN
                    The samples are pseudo-random and based on the Sobol sequence to speed up convergence
    References:     https://iopscience.iop.org/article/10.3847/0004-637X/830/1/31/meta
                    Jorgenson 2022
                    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
    """
    # create a Sobol sequence with dimension d='number or parameters' and length 2^m
    sampler = qmc.Sobol(d=parameters.shape[0], scramble=False)
    draws = sampler.random_base2(m=m)

    # scale the sequence to the parameter range
    l_bounds = [l[0] for l in parameters.Bounds]
    u_bounds = [u[1] for u in parameters.Bounds]
    draws = qmc.scale(draws, l_bounds, u_bounds)
    draws = np.round(draws, 5)

    return draws


class Model(ABM.Sugarscepe):
  def __init__(self):
    super().__init__()
    
  def set_parameters(self, fm_sample):
    """
    Type:        Method
    Description: Overwites the model parameters as specified in the 'parameters' argument
    """
    for firm in self.all_firms:
      for parameter, value in fm_sample.items():
        setattr(firm, parameter, value)

  def run_simulation(self, steps):
    """
    Type: Method
    Description: overwrite run simulaiton in parent class to avoid print statement
    """
    for _ in range(steps):
      self.step()
    

def create_dataset(m, steps=15, save=True):
  """
  Type:        Functon
  Description: Crates the training data for the DNN
  Remark:      Batch size is 2^Batch_size 
  """
  # create a deep copy of the initial model state so model __init__ is only executed once
  simulation = Model()
  with open('../data/model_data/simulation_initial_state.dill', 'wb') as file:
    dill.dump(simulation, file)

  # @TODO change this ugly global variable thing
  draws = create_sample_parameters(df_para, m)

  # create a list of dictionaries; each dictionary corresponds to one sample
  parameters_names = df_para['Name'].tolist()
  samples = [dict(zip(parameters_names, draw)) for draw in draws]
  
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
    sim_copy.run_simulation(steps)

    # get the x vector i.e. the desired summary statistics
    _, _, df_md, _ = sim_copy.datacollector.get_data()
    x.append(df_md[df_md['step']==steps-1].values.flatten().tolist())

    end = timeit.default_timer()
    print(f"\rSimulating step: {i + 1} ({round((i + 1)*100/len(samples), 0)}% complete) time: {round(end-start, 3)}", end="", flush=True)
    
  print("\n")

  # Create a troch tensor from the sample parameter values, theta, and the output vectors, x
  theta = torch.Tensor(draws) 
  x = torch.Tensor(x)

  # Create a torch data set for model training 
  dataset = TensorDataset(theta, x) 

  # Save the dataset if specified, using dill
  if save == True:
    with open('../data/model_data/data_loader.dill', 'wb') as file:
      dill.dump(dataset, file)

  return dataset

# @ df_para / model_runs (replace m) / model_steps

def create_dataloader(m, load=True):
  # if load == true the previously saved dataset is loaded
  # note: the size of the dataset will not necessarily match the size specified in m
  if load == True:
    # To load the dataset from the saved file
    with open('../data/model_data/data_loader.dill', 'rb') as file:
        loaded_dataset = dill.load(file)
    
    dataset = loaded_dataset
  
  # if load == False a new dataset is created in create_dataset
  else:
    dataset = create_dataset(m)

  # Split the dataset into training and testing sets
  total_samples = len(dataset)
  train_size = int(0.8 * total_samples)
  test_size = total_samples - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  
  # Create DataLoader instances for training and testing
  train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False)

  return train_dataloader, test_dataloader

# define hyper parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 1000
I = 4  # input size
O = 13 # output size
eta = 0.001 # learning rate

# @TODO try:
# Batch normalization layer
# Adaptive laerning rate
# Node size
# batch? already done??

# build the network class
class Surrogate(nn.Module):
    def __init__(self):
        super(Surrogate, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(I, 100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, O),
        )

    def forward(self, x):
        x = self.flatten(x)
        Z = self.linear_sigmoid_stack(x)
        return Z

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

      # compute loss and add it to running loss
      loss = loss_fn(pred, x)
      running_loss += loss.item()

    test_losses.append(running_loss / len(test_loader))

  # plot the lossess
  # @TODO make this a function 
  print(f'\r Test MSE: {test_losses}')
  plt.plot(train_losses, label='train_loss')
  plt.plot(test_losses, label='val_loss')
  plt.show()     

# get train and test data as dataloaders
train_loader, test_loader = create_dataloader(4, load=False)

# instantiate network with MSE loss and Adam optimizer
surrogate = Surrogate().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=surrogate.parameters(), lr=eta)

# train the model
train(train_loader, test_loader, surrogate, loss_fn, optimizer, epochs)


if __name__ == "__main__":
    #cProfile.run("create_batch(10)", filename="../data/profile_output.txt", sort='cumulative')
    
    # Create a pstats.Stats object from the profile file
    #profile_stats = pstats.Stats("../data/profile_output.txt")

    # Sort and print the top N entries with the highest cumulative time
    #profile_stats.strip_dirs().sort_stats('cumulative').print_stats(20)

    print('')