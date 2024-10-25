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
        delay = round(random.uniform(0.1, 1.0), 2)  # Delay between 0.1 and 1 ms
        vcpu_usage = random.randint(1, 4)  # vCPU usage between 1 and 4
        network_usage = random.randint(50, 500)  # Network usage between 50 and 500 Mbps
        vnfs.append(VNF(vnf_id, delay, vcpu_usage, network_usage))
    return vnfs


def create_random_slices(num_slices, vnfs, topology):
    slices = []
    for slice_id in range(num_slices):
        # Randomly pick slice types and QoS requirements
        slice_type = random.choice(["uRLLC", "mMTC", "eMBB", "OTHER"])
        latency = random.randint(1, 10)  # Latency in ms
        bandwidth = random.randint(100, 1000)  # Bandwidth in Mbps
        qos = QoS(slice_id, latency, bandwidth)

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
    # Step 1: Create the network topology
    generator = NetworkTopologyGenerator()
    topology = generator.graph
    generator.draw("topology.pdf")

    # Step 2: Create VNFs and network slices
    vnfs = create_random_vnfs(num_vnfs=10)
    slices = create_random_slices(num_slices=3, vnfs=vnfs, topology=topology)

    # Step 3: Create the VNF placement environment
    env = VNFPlacementEnv(topology=topology, slices=slices)

    # Step 4: Initialize the DQN agent
    # state_size = env.state.shape[0]  # Size of the state space
    # action_size = len(vnfs) * len(
    #    topology.nodes
    # )  # Number of possible actions (VNF, Node)
    state_size = len(env.get_state())
    action_size = len(topology.nodes)
    agent = DQNAgent(state_dim=state_size, action_dim=action_size)

    # @TODO> this should be a method --> agent.train()
    # Step 5: Train the DQN agent
    num_episodes = 1000
    # batch_size = 32

    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment for a new episode
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)  # Agent chooses action based on state
            next_state, reward, done, _ = env.step(
                action
            )  # Apply action in the environment
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state  # Move to next state
            total_reward += reward

            # if len(agent.memory) > batch_size:
        agent.replay()  # Train the agent with experience replay

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Step 6: Save the trained model
    trained_file = "dqn_agent.pth"
    agent.save("dqn_agent.pth")
    print("Training completed and model saved.")

    # Step 7: Evaluate the trained agent
    evaluator = Evaluator(topology, slices, trained_file)
    energy_dqn = evaluator.evaluate_model()
    print(f"Energy consumption using DQN: {energy_dqn} Watts")

    # Random, greedy, and round-robin placement for comparison
    energy_random = evaluator.random_vnf_placement()
    energy_greedy = evaluator.greedy_vnf_placement()
    energy_rr = evaluator.round_robin_vnf_placement()

    print(f"Energy consumption (Random): {energy_random} Watts")
    print(f"Energy consumption (Greedy): {energy_greedy} Watts")
    print(f"Energy consumption (Round Robin): {energy_rr} Watts")


#    main()
