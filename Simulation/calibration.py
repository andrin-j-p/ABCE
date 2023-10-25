import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ABM


class Calibrationmodel(ABM.Sugarscepe):
  def __init__(self):
    super().__init__()
  
  def set_parameters(self):
      pass

  def create_batch(self, batch_size):
      pass
      
      


# create iterable for automatic batching, sampling and shuffling
train_loader = DataLoader(train_set, shuffle=True, batch_size=100)
test_loader = DataLoader(test_set, shuffle=False, batch_size=1000)


# build network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


# initialize hyper parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 25
K = 10
O = 10
eta = 0.001
mu = 0.5

# instantiate network
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=eta, momentum=mu)


def train(train_loader, model, loss_fn, optimizer):
    for epoch in range(epochs):
        for xb, tb in train_loader:
            # reset the gradient
            optimizer.zero_grad()

            # Compute prediction and its error
            pred = model(xb)
            loss = loss_fn(pred, tb)
            loss.backward()

            # Perform weight update
            optimizer.step()

        # calculate accuracy using test data set
        correct = 0
        with torch.no_grad():
            for x, t in test_loader:
                pred = model(x)
                # Note that the logits are used rather then the actual probabilities because the canonical order
                # is the same
                correct += (pred.argmax(1) == t).sum().item()
            accuracy = correct/len(test_set)
            print(f"\rloss: {loss:1.7f}  Accuracy: {accuracy:1.7f} Epoch: {epoch}", end="", flush=True)


start = time.time()
train(train_loader, model, loss_fn, optimizer)
end = time.time()
print((end-start)/60)
