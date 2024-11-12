import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        Initialize the deep Q-network with an LSTM for variable-length input sequences.

        :param input_dim: Dimension of the input per node (observation space)
        :param output_dim: Dimension of the output (action space)
        :param hidden_dim: Number of units in the LSTM hidden layer
        """
        super(DeepQNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Assume x is of shape [batch_size, sequence_length, input_dim]
        _, (h_n, _) = self.lstm(x)  # Get the hidden state from LSTM
        x = torch.relu(self.fc1(h_n.squeeze(0)))  # Pass through FC layers
        return self.fc2(x)  # Q-values for each action


class DQNAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        node_ids,
        gamma=0.99,
        epsilon=1.0,
        lr=0.001,
        epsilon_min=0.10,  # 0.15
        epsilon_decay=0.985,  # 0.995
    ):
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, train=True):
        """Choose an action based on epsilon-greedy policy."""
        if train and (np.random.rand() <= self.epsilon):
            return random.choice(valid_actions)

        state = torch.FloatTensor(state).unsqueeze(
            0
        )  # Shape [1, sequence_length, input_dim]
        with torch.no_grad():
            q_values = self.model(state)

        q_values_masked = torch.full_like(q_values, float("-inf"))
        q_values_masked[0, valid_actions] = q_values[0, valid_actions]

        return torch.argmax(q_values_masked).item()

    def replay(self):
        """Train the DQN with experiences sampled from memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(
                0
            )  # Shape [1, sequence_length, input_dim]
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                max_next_q = (
                    torch.max(self.model(next_state)).item() if not done else 0.0
                )
                target = reward + self.gamma * max_next_q

            q_values = self.model(state)
            target_f = q_values.clone()
            target_f[0, action] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
