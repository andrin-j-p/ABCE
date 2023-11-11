import torch
import arviz as az
from matplotlib import pyplot as plt
import torch.nn as nn



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
train_loader, test_loader = create_dataloader(model_runs=4, model_steps=100, batch_size=12, load=True)

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
  for draw in draws:
     draw = torch.Tensor(draw)
     x_pred = surrogate(draw)
     #x_pred = x_pred
     mse = calculate_L2(x_pred, y)
     samples.append((draw, x_pred, mse))

  # keep only the x-predictions most close to the actually observed outcome
  samples = sorted(samples, key=lambda sample : sample[2])

  selected_samples = samples[:int(len(samples) * 0.01)]

  # return the theta values
  selected_thetas = [sample[0].cpu().data.numpy() for sample in selected_samples]

  # plot the posterior densities
  plot_distributions(selected_thetas)

  return min(selected_samples, key=lambda x: x[2])


def plot_distributions(selected_thetas):
  for i, parameter in enumerate(df_para['Name'].tolist()):
    data = np.array([theta[i] for theta in selected_thetas])

    axes = az.plot_dist(data, rug=True, quantiles=[.05, .5, .95])
    fig = axes.get_figure()
    fig.suptitle(f"Density Intervals for {parameter}")

    plt.show()


    quantiles = [np.quantile(data, .05), np.quantile(data, .5), np.quantile(data, .95)] 
    print(quantiles)



test_theta = torch.tensor([[0.8, 0.3, 0.1, 1, 0.5]])
x_pred = surrogate(test_theta)
#print(list(x_pred))
#print(y)
best_guess = rejection_abc(y, surrogate) 
print(best_guess)
