import gymnasium as gym
import numpy as np
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
import time


class VNFPlacementEnv(gym.Env):
    def __init__(self, topology, slices):
        """
        Initialize the VNF placement environment.

        :param topology: NetworkTopologyGenerator object representing the network topology
        :param slices: List of NetworkSlices to instantiate
        """
        super(VNFPlacementEnv, self).__init__()
        self.topology = topology
        self.slices = slices
        self.current_slice_index = 0
        self.current_vnf_index = 0
        self.node_ids = list(self.topology.graph.nodes)
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.topology.graph.nodes))
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(self.topology.graph.nodes), 7), dtype=np.float32
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
        # Precompute valid K-shortest paths from the slice origin to Core nodes
        self.valid_paths = self._compute_k_shortest_paths(k=5)  # Assuming k=5
        return self._get_observation()

    def _compute_k_shortest_paths(self, k=5):
        """Compute the K-shortest paths from the origin RAN node to all Core nodes."""
        valid_paths = []
        current_slice = self.slices[self.current_slice_index]
        origin = current_slice.origin

        # Filter only Core nodes as targets
        core_nodes = [
            node
            for node in self.node_ids
            if self.topology.graph.nodes[node]["type"] == "Core"
        ]

        for target in core_nodes:
            paths = shortest_simple_paths(
                self.topology.graph, source=origin, target=target, weight="latency"
            )
            count = 0
            for path in paths:
                if count >= k:
                    break
                if self._check_path_qos(path):
                    valid_paths.append(path)
                    count += 1
                else:
                    break  # Stop if the path does not meet QoS
        return valid_paths

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

        # Filter nodes only from precomputed valid paths
        nodes_in_paths = set(node for path in self.valid_paths for node in path)

        valid_nodes = [
            node_id
            for node_id in nodes_in_paths
            if self._can_place_vnf_on_node(
                current_vnf,
                node_id,
                current_slice,
                nx.shortest_path(
                    self.topology.graph,
                    source=current_slice.path[-1],
                    target=node_id,
                    weight="latency",
                ),
            )
        ]

        for node_id in valid_nodes:
            node_data = self.topology.graph.nodes[node_id]
            hypothetical_path = nx.shortest_path(
                self.topology.graph,
                source=current_slice.path[-1],
                target=node_id,
                weight="latency",
            )
            # Encode node type as a one-hot vector (RAN, Edge, Transport, Core)
            node_type_one_hot = [0, 0, 0, 0]
            if node_data["type"] == "RAN":
                node_type_one_hot[0] = 1
            elif node_data["type"] == "Edge":
                node_type_one_hot[1] = 1
            elif node_data["type"] == "Transport":
                node_type_one_hot[2] = 1
            elif node_data["type"] == "Core":
                node_type_one_hot[3] = 1

            # Construct the observation vector for the node
            node_observation = [
                node_data["cpu_limit"] - node_data["cpu_usage"],  # Available CPU
                node_data["energy_base"]
                + (
                    node_data["energy_per_vcpu"] * node_data["cpu_usage"]
                ),  # Node's current energy cost
                (
                    -self._calculate_potential_energy(
                        current_slice, node_id, hypothetical_path
                    )
                ),
            ] + node_type_one_hot  # Append one-hot node type

            observations.append(node_observation)
        return np.array(observations)

    def step(self, action):
        """
        Execute one step in the environment based on the selected action.

        :param action: Index in the node_ids list for VNF placement
        :return: Tuple (observation, reward, done, info)
        """
        reward = 0  # No reward if the selected node is not valid!!
        done = False
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]

        node_id = self.node_ids[action]
        node = self.topology.graph.nodes[node_id]

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
        # Check if the node is valid for the VNF placement
        is_valid = self._can_place_vnf_on_node(
            current_vnf, node_id, current_slice, hypothetical_path
        )

        if is_valid:
            # Calculate the energy for this placement
            chosen_energy = self._calculate_potential_energy(
                current_slice, node_id, hypothetical_path
            )

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
                    print(
                        f"Connected graph. It should have a path. Something is wrong!"
                    )
                if self._can_place_vnf_on_node(
                    current_vnf, alt_node_id, current_slice, alt_path
                ):
                    alt_energy = self._calculate_potential_energy(
                        current_slice, alt_node_id, alt_path
                    )
                    if alt_energy < min_energy:
                        min_energy = alt_energy

            # Calculate reward based on chosen vs. minimum energy
            if chosen_energy <= min_energy:
                reward = (
                    50  # Highest reward for selecting the most energy-efficient option
                )
            else:
                # Reward decreases the further the chosen energy is from the minimum
                reward = 50 - (chosen_energy - min_energy) / min_energy * 50
            reward += 20
            # Place the VNF on the selected node (actual placement after decision)
            node["hosted_vnfs"].append(current_vnf)
            part_path = nx.shortest_path(
                self.topology.graph,
                source=(
                    current_slice.path[-1]
                    if current_slice.path
                    else current_slice.origin
                ),
                target=node_id,
                weight="latency",
            )
            current_slice.path = current_slice.path[:-1] + part_path
            node["cpu_usage"] += current_vnf.vcpu_usage

            # Update indices and check if the episode should end
            if self.current_vnf_index == len(current_slice.vnf_list) - 1:
                reward += 500
                self._update_edges(current_slice)
                done = self.current_slice_index == len(self.slices) - 1
                if not done:
                    self.current_slice_index += 1
                    self.current_vnf_index = (
                        1  # Assuming VNF 0 is already placed at origin
                    )
            else:
                self.current_vnf_index += 1
                done = False
        else:
            reward = -500
            # Large penalty for invalid node selection
            # reward = -1000
            # done = False

        return self._get_observation(), reward, done, {}

    def _calculate_potential_energy(self, slice_obj, node_id, hypothetical_path):
        """
        Calculate the potential energy for placing a VNF at a given node without committing to it.

        :param slice_obj: The current network slice object
        :param node_id: The potential node ID where the VNF might be placed
        :return: Total energy consumption for the path if the VNF were placed on node_id
        """
        if len(hypothetical_path) == 0:
            return float("inf")
        # Determine the start node for the path (last node in current slice path or the origin)
        start_node = slice_obj.path[-1] if slice_obj.path else slice_obj.origin
        # Calculate total energy for this hypothetical path
        total_energy = 0

        # Sum the energy for each node and edge along the path
        for i in range(len(hypothetical_path) - 1):
            edge = self.topology.graph[hypothetical_path[i]][hypothetical_path[i + 1]]
            node = self.topology.graph.nodes[hypothetical_path[i]]
            total_energy += (
                node["energy_base"] + node["energy_per_vcpu"] * node["cpu_usage"]
            )
            total_energy += edge["latency"] * edge["link_usage"]

        # Add the energy consumption of the hypothetical placement node
        final_node = self.topology.graph.nodes[node_id]
        total_energy += (
            final_node["energy_base"]
            + final_node["energy_per_vcpu"] * final_node["cpu_usage"]
        )

        return total_energy

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
        if not self._check_latency_feasibility(
            node_id, slice_obj, hypothetical_path
        ) or not self._check_bandwidth_feasibility(
            node_id, slice_obj, hypothetical_path
        ):
            return False
        return True

    def _check_latency_feasibility(self, node_id, slice_obj, path):
        """Check if latency requirements are feasible for the given node and slice."""
        # slice_obj.path[0] should always be the origin!!
        # start_time = time.time()
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return True
        cumulative_latency = nx.path_weight(self.topology.graph, path, weight="latency")
        # Add the delay of each VNF placed in the current path
        for vnf in slice_obj.vnf_list[: self.current_vnf_index + 1]:
            cumulative_latency += vnf.delay
        # elapsed_time = time.time() - start_time
        # print(f"_check_LATENCY_feasibility took {elapsed_time:.6f} seconds")
        return cumulative_latency <= slice_obj.qos_requirement.latency

    def _check_bandwidth_feasibility(self, node_id, slice_obj, path):
        """Check if bandwidth requirements are feasible for the given node and slice."""
        # slice_obj.path[0] should always be the origin!!
        # start_time = time.time()
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

            # print(f"Capacity: {bandwidth_availability}")
            if bandwidth_availability < slice_obj.qos_requirement.bandwidth:
                return False
        # elapsed_time = time.time() - start_time
        # print(f"_check_BANDWIDTH_feasibility took {elapsed_time:.6f} seconds")

        return True

    # @TODO: PAREI AQUI
    def _calculate_path_energy(self, slice_obj):
        """Calculate the energy consumption for the entire slice path."""
        # cumulative_latency = nx.path_weight(self.topology.graph, path, weight="latency")

        path_energy = 0
        for node_id in slice_obj.path:
            node = self.topology.graph.nodes[node_id]
            path_energy += (
                node["energy_base"] + node["energy_per_vcpu"] * node["cpu_usage"]
            )

        # print(f"PATH: {slice_obj.path}")
        edges = self.topology.graph.edges()
        # Add energy costs along each link in the path
        for i in range(0, len(slice_obj.path) - 1):
            edge = edges[[slice_obj.path[i], slice_obj.path[i + 1]]]
            # edge = edges[slice_obj.path[i]][slice_obj.path[i + 1]]
            # edge = self.topology.graph.edges[slice_obj.path[i]][slice_obj.path[i + 1]]
            path_energy += edge["latency"] * edge["link_usage"]

        return -path_energy  # Negative reward for higher energy consumption

    def render(self, mode="human"):
        """Render the environment's current state."""
        print("Rendering environment state (extend for more detail)")
