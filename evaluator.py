from vnf_placement_env import VNFPlacementEnv
import random
import numpy as np


class Evaluator:
    def __init__(self, agent, topology, slices):
        """
        Initialize the evaluator.

        :param agent: Trained DQN agent
        :param topology: Network topology to use for evaluation
        :param slices: List of NetworkSlices to place in the topology
        """
        self.agent = agent
        self.topology = topology
        self.slices = slices

    def evaluate_model(self):
        """Evaluate energy consumption with the DQN agent."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        while not done:
            action = self.agent.act(state)
            state, reward, done, _ = env.step(action)
            total_energy += -reward  # Reward is negative energy, so negate for total
        return total_energy

    def random_vnf_placement(self):
        """Random VNF placement evaluation."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        while not done:
            action = random.choice(range(env.action_space.n))
            state, reward, done, _ = env.step(action)
            total_energy += -reward
        return total_energy

    def greedy_vnf_placement(self):
        """Greedy VNF placement evaluation based on node energy cost."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        while not done:
            action = np.argmin(
                [node["energy_base"] for _, node in env.topology.graph.nodes(data=True)]
            )
            state, reward, done, _ = env.step(action)
            total_energy += -reward
        return total_energy

    def round_robin_vnf_placement(self):
        """Round-robin VNF placement evaluation."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        node_index = 0
        while not done:
            action = node_index % env.action_space.n
            node_index += 1
            state, reward, done, _ = env.step(action)
            total_energy += -reward
        return total_energy
