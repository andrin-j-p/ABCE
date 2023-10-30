import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import ABM


GAMMA = 0.99 # discount rate
BATCH_SIZE = 32 # transitions sampled from the replay buffer
BUFFER_SIZE = 50000 # maximum number of transitions stored before overwrite
MIN_REPLAY_SIZE = 1000 # min smaples collected before training
EPSILON_START = 1 # initial exploration rate
EPSION_START = 0.02 # final exploration rate
EPSION_DECAY = 10000 # iterations over which epsilon decays to its min value
TARGET_UPDATE_FREQ = 1000 # ?


class Environment(ABM.Sugarscepe):
  """
  Type:         Child Class of Sugarscape
  Description:  Mock model instance for training. Extends Sugarscape class
  """
  def __init__(self, owner_id):
    super().__init__()
    self.firm = 0
    self.owner = 0
    self.fm_lst = self.schedule.agents_by_type[ABM.Firm].values()
    self.hh_lst = self.schedule.agents_by_type[ABM.Agent].values()

    # obtain the instance of Firm to be trained
    for firm in self.fm_lst:
      if firm.unique_id == f"f_{owner_id}":
          self.firm = firm

    # obtain the instance of the owner of the Firm 
    for hh in self.hh_lst:
       if hh.unique_id == owner_id:
          self.owner = hh

  # overwrite the price and quantity
  def take_action(self, action):
      self.firm.price_vec[0] = action[0]
      self.firm.quantity_vec[0] = action[1]

  # retrieve the current state
  def get_state(self):
    # get average price on the market the firm operates on 
    total_value = 0
    total_firms = 0
    for firm in self.fm_lst:
      if firm.market == self.owner.village.market:
        total_value += firm.price_vec[0]
        total_firms += 1

    average_price = total_value / total_firms

    # get the total demand the firm faces based on the number of agents for which the firm is the best and second best dealer 
    first_deg_demand = 0
    second_deg_demand = 0
    for hh in self.hh_lst:
      if self.owner == hh.best_dealers[0]:
        first_deg_demand += hh.demand
      if self.owner == hh.best_dealers[1]:
         second_deg_demand += hh.demand

    return (average_price, first_deg_demand, second_deg_demand)
    
       
env = Environment(owner_id='601010102003-101')
print(env.firm)
    
for i in range(10):
  action = np.random.randint(0, 10, size=2)
  step = env.schedule.steps
  print(f"action at step {step}:{action}")
  # otherwise best delaer list etc. are not yet implemented 
  if step > 0:
    env.take_action(action)
  env.step()
  print(env.get_state())

    
# template network to produce instances of Q_Network and Target network
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        # define network architecture (one hidden layer)
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Deep Q-Learning Agent
class DQNAgent:
  def __init__(self, state_size, action_size, epsilon=0.1, gamma=0.9):
      self.state_size = state_size
      self.action_size = action_size
      self.epsilon = epsilon # exploration rate
      self.gamma = gamma
      self.Q_network = Network(state_size, action_size)
      self.T_network = Network(state_size, action_size)
      self.optimizer = optim.Adam(self.Q_network.parameters(), lr=0.001)
      self.memory = []  # Replay memory

  # 'randomly' choose between exploration and exploitation. 
  # the probabilities for exploration and exploitation are governed by epsilon
  def select_action(self, state):
      # Explore
      if random.random() < self.epsilon:
          return random.randint(0, self.action_size - 1)  
      
      # Exploit
      else:
          # select the highest expected Q-value
          with torch.no_grad():
              q_values = self.q_network(state)
              return torch.argmax(q_values).item()  
            
# Initialize the agent and the environment
state_size = 3  # State: total aggregate demand, number of agents ranked #1, number of agents ranked #2
action_size = 2  # Actions: adjust price / quantity
agent = DQNAgent(state_size, action_size)

# Training Loop consisting of 1000 Episodes each with 100 steps
for _ in range(MIN_REPLAY_SIZE):
    state = torch.tensor([10.0, 5.0])  # Initial state (initial price and quantity)
    total_reward = 0