import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
# @TODO decrease exploration rate : more exploitation in later stages


# Define a simple neural network for the Q-function
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
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
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = []  # Replay memory

    def select_action(self, state):
        # randomly choose betwwen exploration and exploitation
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_network(next_state))

            current = self.q_network(state)
            print(current)
            current[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.q_network(state), current)
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Define the environment (exogenous demand)
def get_demand():
    return random.randint(1, 10)

# Initialize the agent and the environment
state_size = 2  # State includes price and quantity
action_size = 2  # Actions: adjust price or adjust quantity
agent = DQNAgent(state_size, action_size)

# Training loop
for episode in range(1000):
    state = torch.tensor([10.0, 5.0])  # Initial state (initial price and quantity)
    total_reward = 0

    for _ in range(100):
        action = agent.select_action(state)
        next_state = state.clone()

        # Adjust price or quantity based on the action
        if action == 0:
            next_state[0] += 1  # Adjust price
        else:
            next_state[1] += 1  # Adjust quantity

        demand = get_demand()
        reward = min(next_state[1], demand) * next_state[0]  # Calculate reward based on sales revenue
        done = False

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    # Experience replay
    agent.experience_replay(batch_size=32)

    if episode % 10 == 0:
        agent.update_target_network()

print("Training completed.")

# Testing loop
state = torch.tensor([10.0, 5.0])  # Initial state (initial price and quantity)
total_reward = 0

for _ in range(100):
    action = agent.select_action(state)
    next_state = state.clone()

    # Adjust price or quantity based on the action
    if action == 0:
        next_state[0] += 1  # Adjust price
    else:
        next_state[1] += 1  # Adjust quantity

    demand = get_demand()
    reward = min(next_state[1], demand) * next_state[0]  # Calculate reward based on sales revenue
    done = False

    state = next_state
    total_reward += reward

print("Total revenue:", total_reward)
