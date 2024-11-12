from collections import defaultdict
import random
import torch
from vnf_placement_env import VNFPlacementEnv
from dql_model import DQNAgent
from evaluator import Evaluator
from network_topology_v3 import (
    NetworkTopologyGenerator,
)
from vnfs_and_slices import NetworkSlice, VNF, QoS
from input_slices import slices
import numpy as np
import matplotlib.pyplot as plt


def plot_training_progress(
    total_rewards,
    average_rewards,
    average_energy_consumption,
    average_latency,
    increments,
):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    increments_range = [i * increment for i in range(1, increments + 1)]

    # Plot total reward per increment
    axs[0, 0].plot(increments_range, total_rewards, color="blue", marker="o")
    axs[0, 0].set_title("Total Reward per Increment")
    axs[0, 0].set_xlabel("Episodes")
    axs[0, 0].set_ylabel("Total Reward")

    # Plot average reward per episode within each increment
    axs[0, 1].plot(increments_range, average_rewards, color="green", marker="o")
    axs[0, 1].set_title("Average Reward per Episode (Increment)")
    axs[0, 1].set_xlabel("Episodes")
    axs[0, 1].set_ylabel("Average Reward")

    # Plot average energy consumption per increment
    axs[1, 0].plot(
        increments_range, average_energy_consumption, color="orange", marker="o"
    )
    axs[1, 0].set_title("Average Energy Consumption per Increment")
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Average Energy Consumption")

    # Plot average latency per increment
    axs[1, 1].plot(increments_range, average_latency, color="purple", marker="o")
    axs[1, 1].set_title("Average Latency per Increment")
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel("Average Latency")

    plt.tight_layout()
    plt.savefig("training_progress.pdf")
    plt.show()


def train_agent(env, agent, episodes, reward_history):
    total_reward = 0
    episode_rewards = []
    avg_energy, avg_latency = [], []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, state, done)
            episode_reward += reward

        agent.replay()  # Replay buffer training step
        total_reward += episode_reward
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {episode_reward}")

        # Track energy and latency from info if available
        avg_energy.append(info.get("total_energy", 0))
        avg_latency.append(info.get("average_latency", 0))

        # Append to reward history
        reward_history[episode].append(episode_reward)

    # Compute metrics
    avg_reward = np.mean(episode_rewards)
    avg_energy = np.mean(avg_energy)
    avg_latency = np.mean(avg_latency)

    return total_reward, avg_reward, avg_energy, avg_latency


# Generate topology and slices
topology = NetworkTopologyGenerator(100)
topology.export_graph_to_pickle("100_nodes_5G_Hierarchical_Topology.pickle")
for slice in slices:
    origin_node = random.choice(
        [n for n, attr in topology.graph.nodes(data=True) if attr["type"] == "RAN"]
    )
    slice.origin = origin_node

# Initialize environment and agent
env = VNFPlacementEnv(topology, slices)
node_ids = env.node_ids
input_dim = env.observation_space.shape[1]
action_dim = env.action_space.n
# print(f"Action Dim: {action_dim}")
agent = DQNAgent(input_dim=input_dim, action_dim=action_dim, node_ids=node_ids)

# Training the agent
total_episodes = 40000
increment = 5000
increments = total_episodes // increment
# I'll be using it later on...
max_attempts = len(topology.graph.nodes) * 2
total_rewards = []
average_rewards = []
average_latency = []
average_energy_consumption = []
reward_history = defaultdict(list)

# Define incremental training procedure
for i in range(increments):
    print(f"Training batch {i+1}/{increments} ({(i+1) * increment} episodes)")
    print(
        "=========================================================================================="
    )
    total_reward, avg_reward, avg_energy, avg_latency = train_agent(
        env, agent, episodes=increment, reward_history=reward_history
    )
    total_rewards.append(total_reward)
    average_rewards.append(avg_reward)
    average_latency.append(avg_latency)
    average_energy_consumption.append(avg_energy)

    # Save model checkpoint after each batch
    torch.save(
        agent.model.state_dict(), f"dqn_model_checkpoint_{(i+1) * increment}.pth"
    )
    print(f"Checkpoint saved for {(i+1) * increment} episodes\n")

plot_training_progress(
    total_rewards,
    average_rewards,
    average_energy_consumption,
    average_latency,
    increments,
)
