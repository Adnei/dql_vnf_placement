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
        epsilon_min=0.15,  # default 0.01
        epsilon_decay=0.995,  # default 0.995
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

    def act(self, state, valid_actions):
        """
        Choose an action based on epsilon-greedy policy, considering only valid actions.

        :param state: Current observation (state) from the environment.
        :param valid_actions: List of valid actions (node indices) as determined by the environment.
        :return: Chosen action (node index).
        """
        # Exploration: pick a random valid action
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        # Exploitation: choose the best action among valid actions
        state = torch.FloatTensor(state).view(1, -1)
        with torch.no_grad():
            q_values = self.model(state)

        # Mask out invalid actions by setting their Q-values to a very low number
        q_values_masked = torch.full_like(q_values, float("-inf"))
        q_values_masked[0, valid_actions] = q_values[0, valid_actions]

        # Select the action with the highest Q-value from valid actions
        return torch.argmax(q_values_masked).item()

    # def act(self, state):
    #     """Choose action based on epsilon-greedy policy."""
    #     # print(f"STATE: {state}")
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_dim)
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         action_values = self.model(state)
    #     return torch.argmax(action_values).item()

    # def act(self, state):
    #     """Choose action based on epsilon-greedy policy."""
    #     if np.random.rand() <= self.epsilon:
    #         # Select a random index within the action space
    #         return random.randint(0, self.action_dim - 1)

    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         action_values = self.model(state)

    #     # Use the max Q-value action
    #     return torch.argmax(action_values).item()  # Return action as index

    # def replay(self):
    #     """Train the DQN with experiences sampled from memory."""
    #     if len(self.memory) < self.batch_size:
    #         return

    #     batch = random.sample(self.memory, self.batch_size)
    #     for state, action, reward, next_state, done in batch:
    #         state = torch.FloatTensor(state)
    #         next_state = torch.FloatTensor(next_state)

    #         # Calculate the target Q-value
    #         target = reward + (
    #             self.gamma * torch.max(self.model(next_state)).item() * (1 - done)
    #         )

    #         # Get the predicted Q-values from the current state
    #         target_f = self.model(state)

    #         # Ensure action is within the bounds of target_f
    #         if action >= len(target_f):
    #             action = len(target_f) - 1  # Adjust action if out of bounds

    #         # Update only the chosen action's Q-value
    #         target_f[action] = target

    #         # Optimize the model
    #         self.optimizer.zero_grad()
    #         loss = F.mse_loss(target_f, self.model(state))
    #         loss.backward()
    #         self.optimizer.step()

    #     # Decay epsilon
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    # def replay(self):
    #     """Train the DQN with experiences sampled from memory."""
    #     if len(self.memory) < self.batch_size:
    #         return

    #     batch = random.sample(self.memory, self.batch_size)
    #     for state, action, reward, next_state, done in batch:
    #         # Flatten or reshape only if necessary
    #         if len(state) != self.model.fc1.in_features:
    #             state = torch.FloatTensor(state).flatten().view(1, -1)
    #         else:
    #             state = torch.FloatTensor(state).view(1, -1)

    #         if len(next_state) != self.model.fc1.in_features:
    #             next_state = torch.FloatTensor(next_state).flatten().view(1, -1)
    #         else:
    #             next_state = torch.FloatTensor(next_state).view(1, -1)

    #         # Calculate the target Q-value
    #         with torch.no_grad():
    #             max_next_q = (
    #                 torch.max(self.model(next_state)).item() if not done else 0.0
    #             )
    #             target = reward + self.gamma * max_next_q

    #         # Get the predicted Q-values for the current state
    #         q_values = self.model(state)

    #         # Select the Q-value for the taken action
    #         q_values[action] = target

    #         # Optimize the model
    #         self.optimizer.zero_grad()
    #         loss = F.mse_loss(q_values, self.model(state))
    #         loss.backward()
    #         self.optimizer.step()

    #     # Decay epsilon
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    def replay(self):
        """Train the DQN with experiences sampled from memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            # Reshape to match input_dim, ensuring compatibility with self.model.fc1
            state = torch.FloatTensor(state).view(1, self.model.fc1.in_features)
            next_state = torch.FloatTensor(next_state).view(
                1, self.model.fc1.in_features
            )

            # Calculate the target Q-value
            with torch.no_grad():
                max_next_q = (
                    torch.max(self.model(next_state)).item() if not done else 0.0
                )
                target = reward + self.gamma * max_next_q

            # Get the predicted Q-values for the current state
            q_values = self.model(state)
            q_values[action] = target  # Only update Q-value for the selected action

            # Optimize the model
            self.optimizer.zero_grad()
            loss = F.mse_loss(q_values, self.model(state))
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
