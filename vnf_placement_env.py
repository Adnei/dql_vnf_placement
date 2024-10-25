import networkx as nx
import numpy as np


class VNFPlacementEnv:
    def __init__(self, topology, slices):
        """
        Initialize the environment with a network topology and network slices.

        :param topology: networkx graph representing the network topology
        :param slices: list of NetworkSlice objects to be placed in the environment
        """
        self.topology = topology
        self.slices = slices
        self.current_slice_index = 0
        self.current_vnf_index = 0
        self.total_energy = 0

    def reset(self):
        """
        Reset the environment, clearing all VNFs and paths from the network.

        :return: Initial state of the environment
        """
        for node in self.topology.nodes:
            self.topology.nodes[node]["hosted_vnfs"] = []
            self.topology.nodes[node]["cpu_usage"] = 0
            self.topology.nodes[node]["link_usage"] = 0
        self.current_slice_index = 0
        self.current_vnf_index = 0
        self.total_energy = 0
        return self.get_state()

    def get_state(self):
        """
        Get the current state of the environment.

        :return: A flattened list of node and slice attributes to represent the state
        """
        state = []
        type_mapping = {"RAN": 0, "Edge": 1, "Transport": 2, "Core": 3}
        for node in self.topology.nodes:
            node_attrs = self.topology.nodes[node]
            node_type = type_mapping[node_attrs["type"]]
            state.extend(
                [
                    node_attrs["cpu_usage"],
                    node_attrs["cpu_limit"],
                    node_attrs["energy_base"],
                    node_attrs["energy_per_vcpu"],
                    node_type,  # RAN, Edge, Transport, Core
                ]
            )
        return state

    def step(self, action):
        """
        Execute a step in the environment by placing a VNF in a specified node (action).

        :param action: node index where the current VNF will be placed
        :return: new state, reward, done flag, and additional info
        """
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]

        if self.can_place_vnf(current_slice, current_vnf, action):
            self.place_vnf(current_vnf, action)
            self.update_path_usage(current_slice, action)
            reward = -self.calculate_total_energy()
        else:
            reward = -float("inf")  # High penalty for invalid placements

        done = (
            self.current_slice_index == len(self.slices) - 1
            and self.current_vnf_index == len(current_slice.vnf_list) - 1
        )

        if not done:
            self.current_vnf_index += 1
            if self.current_vnf_index >= len(current_slice.vnf_list):
                self.current_vnf_index = 0
                self.current_slice_index += 1

        return self.get_state(), reward, done, {}

    # @FIXME!!!!!! What if origin == node ???
    def can_place_vnf(self, network_slice, vnf, node):
        """
        Check if a VNF can be placed on a specific node given QoS and resource constraints.

        :param network_slice: the network slice to which the VNF belongs
        :param vnf: VNF object to be placed
        :param node: node index to check
        :return: True if the VNF can be placed, False otherwise
        """
        node_attrs = self.topology.nodes[node]

        # Check if the node has enough CPU resources
        if node_attrs["cpu_usage"] + vnf.vcpu_usage > node_attrs["cpu_limit"]:
            return False

        # Calculate the path latency and bandwidth requirements
        path = nx.shortest_path(self.topology, network_slice.origin, node)
        print("DEBUG!!!\n")
        print(f"{path}")
        total_latency = sum(
            self.topology[u][v]["latency"] for u, v in zip(path[:-1], path[1:])
        )
        min_bandwidth = min(
            self.topology[u][v]["link_capacity"] - self.topology[u][v]["link_usage"]
            for u, v in zip(path[:-1], path[1:])
        )

        # Check if the QoS requirements of the slice are met
        qos = network_slice.qos_requirement
        if total_latency > qos.latency or min_bandwidth < qos.bandwidth:
            return False

        return True

    def place_vnf(self, vnf, node):
        """
        Place a VNF on a node, updating the node's CPU usage and energy consumption.

        :param vnf: VNF object to be placed
        :param node: node index where the VNF is being placed
        """
        node_attrs = self.topology.nodes[node]
        node_attrs["hosted_vnfs"].append(vnf)
        node_attrs["cpu_usage"] += vnf.vcpu_usage

        # Update the node's energy usage based on new CPU utilization
        additional_energy = vnf.vcpu_usage * node_attrs["energy_per_vcpu"]
        node_attrs["energy_base"] += additional_energy
        self.total_energy += additional_energy

    def update_path_usage(self, network_slice, end_node):
        """
        Update the bandwidth usage along the path from the slice's RAN origin to the selected node.

        :param network_slice: NetworkSlice object containing the QoS requirements
        :param end_node: Node index where the last VNF in the slice path was placed
        """
        path = nx.shortest_path(self.topology, network_slice.origin, end_node)

        for u, v in zip(path[:-1], path[1:]):
            self.topology[u][v]["link_usage"] += network_slice.qos_requirement.bandwidth

    def calculate_total_energy(self):
        """
        Calculate the total energy consumption for the entire network.

        :return: Sum of energy consumption for all nodes in the network
        """
        total_energy = 0
        for node in self.topology.nodes:
            node_attrs = self.topology.nodes[node]
            total_energy += node_attrs["energy_base"]
        return total_energy
