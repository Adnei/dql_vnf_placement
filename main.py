import networkx as nx
import random
import torch
import numpy as np
from network_topology_v3 import NetworkTopologyGenerator

# from network_topology_v2 import NetworkTopologyGenerator_v2
from vnfs_and_slices import VNF, NetworkSlice, QoS
from dql_model import DQNAgent
from vnf_placement_env import VNFPlacementEnv
from evaluator import Evaluator


def create_random_vnfs(num_vnfs):
    vnfs = []
    for vnf_id in range(num_vnfs):
        delay = round(random.uniform(0.1, 0.5), 2)  # Delay between 0.1 and 0.5 ms
        vcpu_usage = random.randint(1, 4)  # vCPU usage between 1 and 4
        # network_usage = random.randint(50, 500)  # Network usage between 50 and 500 Mbps
        vnfs.append(VNF(vnf_id, delay, vcpu_usage))
    return vnfs


def create_random_slices(num_slices, vnfs, topology):
    slices = []
    for slice_id in range(num_slices):
        # Randomly pick slice types and QoS requirements
        slice_type = random.choice(["uRLLC", "mMTC", "eMBB", "OTHER"])
        latency = random.randint(5, 10)  # Latency in ms
        edge_latency = random.choice([None] + list(range(1, 5)))
        bandwidth = random.randint(100, 200)  # Bandwidth in Mbps
        qos = QoS(slice_id, latency, bandwidth, edge_latency)

        # Select origin and path based on the topology (simplified)
        origin_node = random.choice(
            [n for n, attr in topology.nodes(data=True) if attr["type"] == "RAN"]
        )

        core_node = random.choice(
            [n for n, attr in topology.nodes(data=True) if attr["type"] == "Core"]
        )
        vnf_list = random.sample(
            vnfs, random.randint(1, len(vnfs))
        )  # Random VNFs for the slice

        # NetworkSlice constructor initializes it as None
        # slice_path = None  # Not instantiated yet

        slices.append(
            NetworkSlice(slice_id, slice_type, qos, origin_node, core_node, vnf_list)
        )
    return slices


# def main():

if __name__ == "__main__":

    topology = NetworkTopologyGenerator()
    origin_node = random.choice(
        [n for n, attr in topology.graph.nodes(data=True) if attr["type"] == "RAN"]
    )
    # Setup: Define VNFs, Slices, QoS, and Network
    vnfs = [
        VNF(vnf_id=1, vnf_type="RAN", delay=0.5, vcpu_usage=2),
        VNF(vnf_id=3, vnf_type="Transport", delay=0.1, vcpu_usage=1),
        VNF(vnf_id=2, vnf_type="Core", delay=0.5, vcpu_usage=4),
    ]
    qos = QoS(qos_id=1, latency=20, edge_latency=5, bandwidth=100)
    slices = [
        NetworkSlice(
            slice_id=1,
            slice_type="eMBB",
            qos_requirement=qos,
            origin=origin_node,
            vnf_list=vnfs,
        )
    ]

    # Environment and Agent
    # topology.draw()
    env = VNFPlacementEnv(topology, slices)
    agent = DQNAgent(
        state_size=len(topology.graph.nodes), action_size=len(topology.graph.nodes)
    )

    # Train
    episodes = 5000
    for e in range(episodes):
        done = False
        total_reward = 0
        state = env.reset()
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        if len(agent.memory) > 32:
            agent.replay(32)
        print(f"Episode {e}/{episodes} - Total Reward: {total_reward}")

    trained_file = "dqn_agent.pth"
    agent.save("dqn_agent.pth")
    print("Training completed and model saved.")


#    main()
