import random
import numpy as np
from vnf_placement_env import VNFPlacementEnv
from dql_model import DQNAgent


class Evaluator:
    def __init__(self, graph, slices, agent_path):
        """
        Initialize the Evaluator with the network graph, network slices, and path to a trained agent.

        :param graph: networkx graph representing the network topology
        :param slices: list of NetworkSlice objects to be placed
        :param agent_path: file path to the saved DQNAgent model
        """
        self.graph = graph
        self.slices = slices
        self.agent = DQNAgent(
            state_dim=self.get_state_dim(), action_dim=len(graph.nodes)
        )
        self.agent.load(agent_path)

    def get_state_dim(self):
        """
        Calculate the state dimension for the DQNAgent based on the graph and network slice attributes.
        """
        sample_env = VNFPlacementEnv(self.graph, self.slices)
        return len(sample_env.reset())

    def evaluate_model(self):
        """
        Evaluate the trained DQN agent by placing VNFs in the network according to the learned policy.

        :return: Total energy consumption after DQL-based VNF placement
        """
        env = VNFPlacementEnv(self.graph, self.slices)
        state = env.reset()
        done = False
        while not done:
            action = self.agent.act(state, evaluate=True)
            state, reward, done, _ = env.step(action)
        return env.calculate_total_energy()

    def random_vnf_placement(self):
        """
        Place VNFs randomly in the network, respecting resource limits and QoS requirements.

        :return: Total energy consumption after random VNF placement
        """
        env = VNFPlacementEnv(self.graph, self.slices)
        env.reset()
        for slice_ in self.slices:
            for vnf in slice_.vnf_list:
                placed = False
                while not placed:
                    node = random.choice(list(self.graph.nodes))
                    if env.can_place_vnf(slice_, vnf, node):
                        env.place_vnf(vnf, node)
                        placed = True
        return env.calculate_total_energy()

    def greedy_vnf_placement(self):
        """
        Place VNFs in the network in a greedy manner, selecting nodes with the least energy consumption impact.

        :return: Total energy consumption after greedy VNF placement
        """
        env = VNFPlacementEnv(self.graph, self.slices)
        env.reset()
        for slice_ in self.slices:
            for vnf in slice_.vnf_list:
                min_energy_node = None
                min_energy = float("inf")
                for node in self.graph.nodes:
                    if env.can_place_vnf(slice_, vnf, node):
                        projected_energy = env.calculate_node_energy(node, vnf)
                        if projected_energy < min_energy:
                            min_energy = projected_energy
                            min_energy_node = node
                if min_energy_node is not None:
                    env.place_vnf(vnf, min_energy_node)
        return env.calculate_total_energy()

    def round_robin_vnf_placement(self):
        """
        Place VNFs in a round-robin fashion across nodes, respecting resource limits and QoS requirements.

        :return: Total energy consumption after round-robin VNF placement
        """
        env = VNFPlacementEnv(self.graph, self.slices)
        env.reset()
        nodes = list(self.graph.nodes)
        node_count = len(nodes)
        for i, slice_ in enumerate(self.slices):
            for j, vnf in enumerate(slice_.vnf_list):
                placed = False
                while not placed:
                    node = nodes[(i + j) % node_count]
                    if env.can_place_vnf(slice_, vnf, node):
                        env.place_vnf(vnf, node)
                        placed = True
        return env.calculate_total_energy()
