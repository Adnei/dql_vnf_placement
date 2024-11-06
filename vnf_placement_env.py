import gymnasium as gym
import numpy as np
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
import time


class VNFPlacementEnv(gym.Env):
    def __init__(self, topology, slices):
        super(VNFPlacementEnv, self).__init__()
        self.topology = topology
        self.slices = slices
        self.current_slice_index = 0
        self.current_vnf_index = 0
        self.node_ids = list(self.topology.graph.nodes)
        self.features_per_node = 3
        self.cached_paths = {}
        self.action_space = gym.spaces.Discrete(len(self.node_ids))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.node_ids), self.features_per_node),
            dtype=np.float32,
        )

    def reset(self):
        """Reset the environment for a new episode."""
        self.current_slice_index = 0
        self.current_vnf_index = 0

        # Clean network topology
        for u, v, attrs in self.topology.graph.edges(data=True):
            attrs["link_usage"] = 0
        for node_id, node_data in self.topology.graph.nodes(data=True):
            node_data["cpu_usage"] = 0
            node_data["hosted_vnfs"] = []

        # We place the first VNF to the origin node.
        for slice_obj in self.slices:
            slice_obj.path = [slice_obj.origin]
            self.topology.graph.nodes[slice_obj.origin]["hosted_vnfs"] = (
                slice_obj.vnf_list[0]
            )
            self.topology.graph.nodes[slice_obj.origin]["cpu_usage"] = (
                slice_obj.vnf_list[0].vcpu_usage
            )
        self.current_vnf_index = 1
        return self._get_observation()

    def _check_path_qos(self, path):
        """Check if a path meets QoS requirements (latency and bandwidth)."""
        total_latency = nx.path_weight(self.topology.graph, path, weight="latency")
        if (
            total_latency
            > self.slices[self.current_slice_index].qos_requirement.latency
        ):
            return False
        for i in range(len(path) - 1):
            edge_data = self.topology.graph[path[i]][path[i + 1]]
            if (
                edge_data["link_capacity"] - edge_data["link_usage"]
                < self.slices[self.current_slice_index].qos_requirement.bandwidth
            ):
                return False
        return True

    def _get_observation(self):
        """Generate an observation vector based on valid nodes in the precomputed K-shortest paths."""
        observations = []
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]

        valid_actions = self.get_valid_actions()
        # print(f"VNF to be placed: {current_vnf}")
        # print(f"Valid Nodes: {valid_nodes} \n")

        for node_id in valid_actions:
            node_data = self.topology.graph.nodes[node_id]
            hypothetical_path = nx.shortest_path(
                self.topology.graph,
                source=current_slice.path[-1],
                target=node_id,
                weight="latency",
            )

            total_latency, _ = self._check_latency_feasibility(
                node_id, current_slice, hypothetical_path
            )

            # Construct the observation vector for the node
            node_observation = [
                node_data["cpu_limit"] - node_data["cpu_usage"],  # Available CPU
                -self._calculate_path_energy(hypothetical_path, place_vnf=current_vnf),
                -total_latency,
            ]

            observations.append(node_observation)
        return np.array(observations)

    def get_valid_actions(self):
        """Generate a list of valid actions (node indices) for the current VNF placement."""
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]

        # Filter nodes matching current vnf type
        nodes_matching_vnf_type = [
            node_id
            for node_id, data in self.topology.graph.nodes(data=True)
            if data["type"] == current_vnf.vnf_type
        ]

        valid_actions = [
            node_id
            for node_id in nodes_matching_vnf_type
            if self._can_place_vnf_on_node(
                current_vnf,
                node_id,
                current_slice,
                self._get_hypothetical_path(current_slice, node_id),
            )
        ]

        return valid_actions

    def step(self, action):
        done = False
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            print(f"ACTION MASK NOT RIGHT!!! ACTION SHOULD ALWAYS BE VALID!!!!")
            return self._get_observation(), -1000, False, {}

        node_id = self.node_ids[action]
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]
        hypothetical_path = self._get_hypothetical_path(current_slice, node_id)
        info = {}

        node = self.topology.graph.nodes[node_id]
        node["hosted_vnfs"].append(current_vnf)
        current_slice.path.extend(hypothetical_path[1:])
        node["cpu_usage"] += current_vnf.vcpu_usage
        chosen_energy = self._calculate_path_energy(current_slice.path)

        min_energy = self._find_minimum_energy_placement(current_slice, current_vnf)
        reward = 50 + (50 - (chosen_energy - min_energy) / min_energy * 50)

        if self.current_vnf_index == len(current_slice.vnf_list) - 1:
            self._update_edges(current_slice)
            done = self.current_slice_index == len(self.slices) - 1
            if not done:
                self.current_slice_index += 1
                self.current_vnf_index = 1  # Assuming VNF 0 is already placed at origin
                reward += 200
        else:
            self.current_vnf_index += 1
            done = False

        if done:
            info["total_energy"] = chosen_energy
            info["e2e_latency"], _ = self._check_latency_feasibility(
                node_id, current_slice, current_slice.path
            )
            bandwidth_availability = np.inf
            for node_idx in range(0, len(current_slice.path) - 1):
                bandwidth_availability = min(
                    (
                        self.topology.graph.edges[
                            current_slice.path[node_idx],
                            current_slice.path[node_idx + 1],
                        ]["link_capacity"]
                        - self.topology.graph.edges[
                            current_slice.path[node_idx],
                            current_slice.path[node_idx + 1],
                        ]["link_usage"]
                    ),
                    bandwidth_availability,
                )

            info["remaining_bandwidth_in_slice_path"] = bandwidth_availability

        return self._get_observation(), reward, done, info

    def _update_edges(self, slice_obj):
        """Update bandwidth usage and availability after the slice is deployed"""
        edges = self.topology.graph.edges()
        for i in range(0, len(slice_obj.path) - 1):
            edge = edges[[slice_obj.path[i], slice_obj.path[i + 1]]]
            edge["link_usage"] += slice_obj.qos_requirement.bandwidth

    def _can_place_vnf_on_node(self, vnf, node_id, slice_obj, hypothetical_path):
        """Check if a VNF can be placed on a specific node considering resources and QoS constraints."""
        if len(hypothetical_path) == 0:
            return False
        node = self.topology.graph.nodes[node_id]
        if (
            node["type"] != vnf.vnf_type
            or node["cpu_usage"] + vnf.vcpu_usage > node["cpu_limit"]
        ):
            return False
        path_latency, feasible = self._check_latency_feasibility(
            node_id, slice_obj, hypothetical_path
        )
        if not feasible or not self._check_bandwidth_feasibility(
            node_id, slice_obj, hypothetical_path
        ):
            return False
        return True

    def _check_latency_feasibility(self, node_id, slice_obj, path):
        """Check if latency requirements are feasible for the given node and slice."""
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return 0, True
        cumulative_latency = nx.path_weight(self.topology.graph, path, weight="latency")
        # Add the delay of each VNF placed in the current path
        for vnf in slice_obj.vnf_list[: self.current_vnf_index]:
            cumulative_latency += vnf.delay
        return (
            cumulative_latency,
            cumulative_latency <= slice_obj.qos_requirement.latency,
        )

    def _check_bandwidth_feasibility(self, node_id, slice_obj, path):
        """Check if bandwidth requirements are feasible for the given node and slice."""
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return True

        for node_idx in range(0, len(path) - 1):
            bandwidth_availability = (
                self.topology.graph.edges[path[node_idx], path[node_idx + 1]][
                    "link_capacity"
                ]
                - self.topology.graph.edges[path[node_idx], path[node_idx + 1]][
                    "link_usage"
                ]
            )

            if bandwidth_availability < slice_obj.qos_requirement.bandwidth:
                return False
        return True

    def _calculate_path_energy(self, path, place_vnf=None):
        """
        Calculate the energy consumption for the entire slice path.
        place_vnf=None or <VNF object>
        if place_vnf is a VNF then we simulate placing the VNF in the last node in path (path[-1])

        Returns the total energy consumption of the path
        """

        path_energy = 0

        for node_idx in range(len(path)):
            node = self.topology.graph.nodes[path[node_idx]]
            alpha = 1
            # Simulates VNF placement on last node of the path
            if place_vnf != None and node_idx == len(path) - 1:
                alpha = place_vnf.vcpu_usage
            path_energy += node["energy_base"] + node["energy_per_vcpu"] * (
                node["cpu_usage"] * alpha
            )

        return path_energy

    def _get_hypothetical_path(self, current_slice, node_id):
        hypothetical_path = []
        try:
            hypothetical_path = nx.shortest_path(
                self.topology.graph,
                source=current_slice.path[-1],
                target=node_id,
                weight="latency",
            )
        except nx.NetworkXNoPath:
            print(f"Connected graph. It should have a path. Something is wrong!")
        return hypothetical_path

    def _find_minimum_energy_placement(self, current_slice, current_vnf):
        # Calculate the minimum energy among all possible valid placements for this VNF
        min_energy = float("inf")
        for alt_node_id in self.node_ids:
            alt_node = self.topology.graph.nodes[alt_node_id]

            alt_path = []
            try:
                alt_path = nx.shortest_path(
                    self.topology.graph,
                    source=current_slice.path[-1],
                    target=alt_node_id,
                    weight="latency",
                )
            except nx.NetworkXNoPath:
                print(f"Connected graph. It should have a path. Something is wrong!")
            if self._can_place_vnf_on_node(
                current_vnf, alt_node_id, current_slice, alt_path
            ):
                alt_energy = self._calculate_path_energy(
                    alt_path, place_vnf=current_vnf
                )
                if alt_energy < min_energy:
                    min_energy = alt_energy
        return min_energy

    def render(self, mode="human"):
        """Render the environment's current state."""
        print("Rendering environment state (extend for more detail)")
