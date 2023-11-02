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
import dill

# Parameters to be calibrated
# theta: probability for price change
# nu:    rate for price change
# phi_l: lower inventory rate 
# phi_u: uper inventory rate 
data = {'Type': ['fm', 'fm', 'fm', 'fm'], 
        'Name': ['theta', 'nu', 'phi_l', 'phi_u'],
        'Bounds': [(0,1), (0.05, 0.5), (0.05, 0.5), (0.5, 1)]} 
  
# Create DataFrame 
df_para = pd.DataFrame(data) 

# Summary statistics to be validated against
calibration_features = ['average_stock', 'unemployment_rate', 'average_income']
y = torch.tensor([8633.905329, 0.320321, 1784.718346]) # after 100 steps @ DETELE solve this using a regular model run 


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


def create_dataset(model_runs, model_steps):
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
    sim_copy.run_simulation(model_steps)

    # get the x vector i.e. the desired summary statistics
    df_sm = sim_copy.datacollector.get_data()
    df_sm = df_sm[df_sm['step']==model_steps-1]
    x.append(df_sm[calibration_features].values.flatten().tolist())
    
    end = timeit.default_timer()
    print(f"\rSimulating step: {i + 1} ({round((i + 1)*100/len(samples), 0)}% complete) time: {round(end-start, 3)}", end="", flush=True)
    
  print("\n")

  # Create a troch tensor from the sample parameter values, theta, and the output vectors, x
  theta = torch.Tensor(draws) 
  x = torch.Tensor(x)

  # Create a torch data set for model training 
  dataset = TensorDataset(theta, x) 

  # Save the dataset as a file
  with open('../data/model_data/data_loader0.dill', 'wb') as file:
    dill.dump(dataset, file)

  return dataset


def create_dataloader(model_runs, model_steps, batch_size, load):
  """
  Type:        Function
  Description: Creates a pytorch dataloader from a dataset 
               used in the training loop
  """
  # if load == true the previously saved dataset is loaded
  # note: the size of the dataset will not necessarily match the size specified in m
  if load == True:
    # To load the dataset from the saved file
    with open('../data/model_data/data_loader.dill', 'rb') as file:
        loaded_dataset = dill.load(file)
    
    dataset = loaded_dataset
  
  # if load == False a new dataset is created in 'create_dataset()'
  else:
    dataset = create_dataset(model_runs, model_steps)

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

epochs = 100
I = df_para.shape[0]  # input size
O = len(y) # output size
eta = 0.001 # learning rate

def hyperparameter_tuning():
  """
  Type:        Helper Function 
  Description: For surrogate model selection
  References:  https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
  """
  # @TODO try:
  # Batch normalization layer
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
            nn.Dropout(0.2), 
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
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
  print(f'\r Test MSE: {test_losses}')
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
train_loader, test_loader = create_dataloader(model_runs=7, model_steps = 100, batch_size=12, load=True)

# Instantiate network with MSE loss and Adam optimizer
surrogate = Surrogate().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=surrogate.parameters(), lr=eta)

# Train the surrogate model
train(train_loader, test_loader, surrogate, loss_fn, optimizer, epochs)

def calculate_MSE(x, y):
   """
   Type:        Helper function 
   Description: Calculates the mean squared loss btw. two vectors
   """
   return torch.mean(torch.square(x - y)).item()


def rejection_abc(y, surrogate):
  """
  Type:        Function 
  Description: Main function to conduct Baysian inference 
  """
  # create 2^17 = 131 072 theta values
  draws = create_sample_parameters(df_para, m=17)    
  
  # for each theta, predict the x vectors using the surrogate and calculate the corresponding mse
  samples = []
  for draw in draws:
     draw = torch.Tensor(draw)
     x_pred = surrogate(draw)
     #x_pred = x_pred
     mse = calculate_MSE(x_pred, y)
     samples.append((draw, x_pred, mse))

  # keep only the x-predictions most close to the actually observed outcome
  samples = sorted(samples, key=lambda sample : sample[2])

  selected_samples = samples[:int(len(samples) * 0.6)]

  # return the theta values
  selected_thetas = [sample[0].cpu().data.numpy()[1] for sample in selected_samples]
  plot_distributions(selected_thetas)
  return selected_thetas


def plot_distributions(selected_thetas):
  posterior_quantiles = np.percentile(selected_thetas, [2.5, 97.5])  # 95% credible interval
  plt.hist(selected_thetas, bins=30, density=True, alpha=0.5, color='b')
  plt.fill_betweenx([0, 1], posterior_quantiles[0], posterior_quantiles[1], alpha=0.3, color='red')
  plt.xlabel(f"Parameter")
  plt.ylabel("Posterior Density")
  plt.title("Posterior Distribution with 95% Credible Interval")
  plt.show()

test_theta = torch.tensor([[0.8, 0.3, 0.1, 1]])
x_pred = surrogate(test_theta)
print(list(x_pred))
print(y)
rejection_abc(y, surrogate) 

