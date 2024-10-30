import torch
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        Initialize the deep Q-network.

        :param input_dim: Dimension of the input (observation space)
        :param output_dim: Dimension of the output (action space)
        :param hidden_dim: Number of units in the hidden layers
        """
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor
        :return: Output Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        node_ids,
        gamma=0.99,
        epsilon=1.0,
        lr=0.001,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        """
        Initialize the DQN agent.

        :param input_dim: Dimension of the observation space
        :param action_dim: Dimension of the action space
        :param gamma: Discount factor
        :param epsilon: Exploration rate
        :param lr: Learning rate
        :param epsilon_min: Minimum exploration rate
        :param epsilon_decay: Rate of decay for epsilon
        """
        self.action_dim = action_dim
        self.node_ids = node_ids
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.batch_size = 64

        # Initialize DQN and optimizer
        self.model = DeepQNetwork(input_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     """Choose action based on epsilon-greedy policy."""
    #     # print(f"STATE: {state}")
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_dim)
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         action_values = self.model(state)
    #     return torch.argmax(action_values).item()

    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            # Select a random node from the valid node IDs
            return random.choice(self.node_ids)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)

        # Use the max Q-value action
        action_index = torch.argmax(action_values).item()
        return self.node_ids[
            action_index % len(self.node_ids)
        ]  # Map index to actual node ID

    def replay(self):
        """Train the DQN with experiences sampled from memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward + (
                self.gamma * torch.max(self.model(next_state)).item() * (1 - done)
            )
            target_f = self.model(state)
            target_f[action] = target

            # Optimize the model
            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
