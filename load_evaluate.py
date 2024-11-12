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

# Generate topology and slices
# topology = NetworkTopologyGenerator(from_file="5G_Hierarchical_Topology.pickle")
topology = NetworkTopologyGenerator(
    from_file="100_nodes_5G_Hierarchical_Topology.pickle"
)
# 100_nodes_5G_Hierarchical_Topology.pickle
# topology.draw()

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

agent.model.load_state_dict(torch.load("dqn_model.pth", weights_only=True))

print("\n\n")
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

# Additional comparison of DQN variance against other methods
energy_variance = np.var(
    [
        dqn_results["total_energy"],
        greedy_results["total_energy"],
        random_results["total_energy"],
        round_robin_results["total_energy"],
    ]
)
print(f"Variance in total energy consumption across methods: {energy_variance}")
