import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from scipy.stats import qmc
import ABM
import time
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
    samples = sampler.random_base2(m=m)

    # scale the sequence to the parameter range
    l_bounds = [l[0] for l in parameters.Bounds]
    u_bounds = [u[1] for u in parameters.Bounds]
    samples = qmc.scale(samples, l_bounds, u_bounds)

    return samples

class Model(ABM.Sugarscepe):
  def __init__(self):
    super().__init__()
    print("Model innit was called")
  
  def set_parameters(self, fm_sample):
    """
    Type:        Method
    Description: Overwites the model parameters as specified in the 'parameters' argument
    """

    for firm in self.all_firms:
      for parameter, value in fm_sample.items():
        setattr(firm, parameter, value)
    

def create_dataloader(batch_size):
  """
  Type:        Functon
  Description: Crates the training data for the DNN
  Remark:      Batch size is 2^Batch_size 
  """
  # create a copy of the initialized model
  # this way the model is only initalized once 
  simulation = Model()

  parameter_values = create_sample_parameters(df_para, batch_size)
  parameters_names = df_para['Name'].tolist()
  
  x = []
  for values in parameter_values:
    simpulation_copy = simulation
    sample = dict(zip(parameters_names, values))
    simulation.set_parameters(sample)
    simpulation_copy.run_simulation(10)
    _, _, df_md, _ = simulation.datacollector.get_data()
    x.append(df_md[df_md['step']==8].values.flatten().tolist())

  theta = torch.Tensor(parameter_values) # transform to torch tensor
  x = torch.Tensor(x)
  print(x.size())
  print(theta.size())

  # create data set
  my_dataset = TensorDataset(theta, x) 
  # create data loader
  dataloader = DataLoader(my_dataset, shuffle=True, batch_size=2)

  return dataloader, []


# define hyper parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 25
K = 10
O = 10
eta = 0.001
mu = 0.5

# build network class
class Surrogate(nn.Module):
    def __init__(self):
        super(Surrogate, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, K),
            nn.Sigmoid(),
            nn.Linear(K, O),
        )

    def forward(self, x):
        x = self.flatten(x)
        Z = self.linear_sigmoid_stack(x)
        return Z

def train(train_loader, model, loss_fn, optimizer):
    for epoch in range(epochs):
        for theta, x in train_loader:
            # reset the gradient
            optimizer.zero_grad()

            # Compute prediction and its error
            pred = model(theta)
            loss = loss_fn(pred, x)
            loss.backward()

            # Perform weight update
            optimizer.step()
            print(loss.detach().numpy())


# instantiate network with MSE loss, @Adam optimizer
train_loader, test_loader = create_dataloader(2)
surrogate = Surrogate().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=surrogate.parameters(), lr=eta)

start = time.time()
train(train_loader, surrogate, loss_fn, optimizer)
end = time.time()
print((end-start)/60)


#train(train_loader, model, loss_fn, optimizer)


if __name__ == "__main__":
    #cProfile.run("create_batch(10)", filename="../data/profile_output.txt", sort='cumulative')
    
    # Create a pstats.Stats object from the profile file
    #profile_stats = pstats.Stats("../data/profile_output.txt")

    # Sort and print the top N entries with the highest cumulative time
    #profile_stats.strip_dirs().sort_stats('cumulative').print_stats(20)
    print('')