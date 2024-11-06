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

# Generate topology and slices
topology = NetworkTopologyGenerator(50)

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
episodes = 1000
max_attempts = len(topology.graph.nodes) * 2
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    for attempt in range(max_attempts):
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        # print(f"STARTING action: {action}")
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    agent.replay()  # Train on memory replay after each episode
    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_model.pth")


print("\n\n")
# Evaluate the model
evaluator = Evaluator(agent, topology, slices)

# Evaluate the DQN Agent
dqn_results = evaluator.evaluate_model()
print("DQN Agent:", dqn_results)

# Evaluate Baselines
random_results = evaluator.random_vnf_placement()
print("Random Placement:", random_results)

greedy_results = evaluator.greedy_vnf_placement()
print("Greedy Placement:", greedy_results)

round_robin_results = evaluator.round_robin_vnf_placement()
print("Round Robin Placement:", round_robin_results)
