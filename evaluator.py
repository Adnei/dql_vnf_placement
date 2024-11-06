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
        """Evaluate the DQN agent's energy consumption and QoS compliance."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        total_latency = 0
        total_bandwidth = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action = self.agent.act(state, valid_actions)
            state, reward, done, info = env.step(action)
            total_energy += -reward
            total_latency += info.get("latency", 0)
            total_bandwidth += info.get("bandwidth", 0)

        return {
            "total_energy": total_energy,
            "average_latency": total_latency / len(self.slices),
            "average_bandwidth": total_bandwidth / len(self.slices),
        }

    def random_vnf_placement(self):
        """Random VNF placement evaluation."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        total_latency = 0
        total_bandwidth = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions)
            state, reward, done, info = env.step(action)
            total_energy += -reward
            total_latency += info.get("latency", 0)
            total_bandwidth += info.get("bandwidth", 0)

        return {
            "total_energy": total_energy,
            "average_latency": total_latency / len(self.slices),
            "average_bandwidth": total_bandwidth / len(self.slices),
        }

    def greedy_vnf_placement(self):
        """Greedy VNF placement evaluation based on node energy cost."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        total_latency = 0
        total_bandwidth = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action = min(
                valid_actions, key=lambda a: env.topology.graph.nodes[a]["energy_base"]
            )
            state, reward, done, info = env.step(action)
            total_energy += -reward
            total_latency += info.get("latency", 0)
            total_bandwidth += info.get("bandwidth", 0)

        return {
            "total_energy": total_energy,
            "average_latency": total_latency / len(self.slices),
            "average_bandwidth": total_bandwidth / len(self.slices),
        }

    def round_robin_vnf_placement(self):
        """Round-robin VNF placement evaluation."""
        env = VNFPlacementEnv(self.topology, self.slices)
        state = env.reset()
        done = False
        total_energy = 0
        total_latency = 0
        total_bandwidth = 0
        node_index = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action = valid_actions[node_index % len(valid_actions)]
            node_index += 1
            state, reward, done, info = env.step(action)
            total_energy += -reward
            total_latency += info.get("latency", 0)
            total_bandwidth += info.get("bandwidth", 0)

        return {
            "total_energy": total_energy,
            "average_latency": total_latency / len(self.slices),
            "average_bandwidth": total_bandwidth / len(self.slices),
        }
