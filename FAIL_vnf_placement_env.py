import gymnasium as gym
import numpy as np
import networkx as nx
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
            low=0, high=1, shape=(len(self.topology.graph.nodes), 8), dtype=np.float32
        )
        self.cached_paths = {}

    def reset(self):
        """Reset the environment for a new episode."""
        self.current_slice_index = 0
        self.current_vnf_index = 0
        for slice_obj in self.slices:
            slice_obj.path = [slice_obj.origin]

        # Cleaning network topology
        # Edges -

        for u, v, attrs in self.topology.graph.edges(data=True):
            attrs["link_usage"] = 0

        # Node attrs
        for node_id, node_data in self.topology.graph.nodes(data=True):
            node_data["cpu_usage"] = 0
            node_data["hosted_vnfs"] = []

        return self._get_observation()

    def _get_observation(self):
        """Generate an observation vector based on current network state and slice QoS requirements."""
        observations = []
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]
        for node_id, node_data in self.topology.graph.nodes(data=True):
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
                # return False
            is_valid = int(
                self._can_place_vnf_on_node(
                    current_vnf, node_id, current_slice, hypothetical_path
                )
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
                is_valid,
                (
                    -self._calculate_potential_energy(
                        current_slice, node_id, hypothetical_path
                    )
                    if is_valid
                    else -np.inf  # should be positive. But we could also work with the potential reward
                ),
            ] + node_type_one_hot  # Append one-hot node type

            observations.append(node_observation)
        return np.array(observations)

    def step(self, action):
        """
        Execute one step in the environment based on the selected action,
        but restrict actions to valid nodes using a validity mask.

        :param action: Index of a valid node in the list of valid actions
        :return: Tuple (observation, reward, done, info)
        """
        # Get current slice and VNF details
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]

        # Generate a list of valid actions based on current VNF requirements
        valid_actions = self._get_valid_actions()

        # If no valid actions are available, end the episode with a penalty
        if not valid_actions:
            reward = -1000  # Large penalty for being unable to place the VNF
            done = True
            return self._get_observation(), reward, done, {}

        # If the chosen action is invalid, penalize and return early
        if action not in valid_actions:
            reward = -1000  # High penalty for selecting an invalid action
            done = False
            return self._get_observation(), reward, done, {}

        # Map the action to the actual node ID
        node_id = self.node_ids[action]
        node = self.topology.graph.nodes[node_id]

        # Place the VNF on the selected valid node
        node["hosted_vnfs"].append(current_vnf)
        hypothetical_path = self._get_hypothetical_path(current_slice, node_id)
        chosen_energy = self._calculate_potential_energy(
            current_slice, node_id, hypothetical_path
        )  # Fix: Pass hypothetical_path

        # Reward for energy-efficient placement
        min_energy = self._find_minimum_energy_placement(current_slice, current_vnf)
        reward = 50 + (50 - (chosen_energy - min_energy) / min_energy * 50)

        # Update the current slice path with the chosen valid node
        part_path = hypothetical_path
        current_slice.path.extend(part_path[1:])  # Avoid duplicating the start node
        node["cpu_usage"] += current_vnf.vcpu_usage

        # Move to the next VNF in the slice
        if self.current_vnf_index == len(current_slice.vnf_list) - 1:
            # End of slice
            self._update_edges(current_slice)
            done = self.current_slice_index == len(self.slices) - 1
            if not done:
                self.current_slice_index += 1
                self.current_vnf_index = 0
                reward += 200  # Reward for successfully completing a slice
        else:
            self.current_vnf_index += 1
            done = False

        return self._get_observation(), reward, done, {}

    def _find_minimum_energy_placement(self, current_slice, current_vnf):
        """
        Find the minimum energy consumption for a valid placement of the current VNF.

        :param current_slice: Current network slice being processed
        :param current_vnf: The VNF being placed
        :return: Minimum energy consumption among all valid nodes for this VNF
        """
        min_energy = float("inf")
        for alt_node_id in self.node_ids:
            if self._can_place_vnf_on_node(
                current_vnf,
                alt_node_id,
                current_slice,
                self._get_hypothetical_path(current_slice, alt_node_id),
            ):
                alt_hypothetical_path = self._get_hypothetical_path(
                    current_slice, alt_node_id
                )
                alt_energy = self._calculate_potential_energy(
                    current_slice, alt_node_id, alt_hypothetical_path
                )  # Fix: Pass hypothetical_path
                if alt_energy < min_energy:
                    min_energy = alt_energy
        return min_energy

    def _get_valid_actions(self):
        """Generate a list of valid actions (node indices) for the current VNF placement."""
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]
        valid_actions = []

        for idx, node_id in enumerate(self.node_ids):
            node_data = self.topology.graph.nodes[node_id]
            hypothetical_path = self._get_hypothetical_path(current_slice, node_id)
            if self._can_place_vnf_on_node(
                current_vnf, node_id, current_slice, hypothetical_path
            ):
                valid_actions.append(idx)  # Add valid node index to the list

        return valid_actions

    def _get_hypothetical_path(self, slice_obj, node_id):
        """
        Retrieve or calculate the hypothetical path to the node.
        """
        start_node = slice_obj.path[-1] if slice_obj.path else slice_obj.origin
        if (start_node, node_id) not in self.cached_paths:
            self.cached_paths[(start_node, node_id)] = nx.shortest_path(
                self.topology.graph, source=start_node, target=node_id, weight="latency"
            )
        return self.cached_paths[(start_node, node_id)]

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
