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
            low=0, high=1, shape=(len(self.topology.graph.nodes), 4), dtype=np.float32
        )

    def reset(self):
        """Reset the environment for a new episode."""
        self.current_slice_index = 0
        self.current_vnf_index = 0
        for slice_obj in self.slices:
            slice_obj.path = [slice_obj.origin]
        return self._get_observation()

    def _get_observation(self):
        """Generate an observation based on current network state and slice QoS requirements."""
        current_slice = self.slices[self.current_slice_index]
        observations = []

        for node_id, node_data in self.topology.graph.nodes(data=True):
            node_observation = [
                node_data["cpu_usage"] / node_data["cpu_limit"],  # CPU usage ratio
                node_data["energy_base"]
                + node_data["energy_per_vcpu"] * node_data["cpu_usage"],  # Node energy
                self._check_latency_feasibility(
                    node_id, current_slice
                ),  # Latency feasibility
                self._check_bandwidth_feasibility(
                    node_id, current_slice
                ),  # Bandwidth feasibility
            ]
            observations.append(node_observation)

        # observations size 50 (len(topology.graph.nodes()))
        # print(len(np.array(observations)))
        return np.array(observations)

    def step(self, action):
        """
        Execute one step in the environment based on the selected action.

        :param action: Selected node for VNF placement
        :return: Tuple (observation, reward, done, info)
        """
        current_slice = self.slices[self.current_slice_index]
        current_vnf = current_slice.vnf_list[self.current_vnf_index]

        node = self.topology.graph.nodes[action]
        # Track number of attempts to place the current VNF
        max_attempts = len(self.topology.graph.nodes)  # Check all nodes at most once
        attempts = 0
        while attempts < max_attempts:
            # Place VNF if QoS and resource requirements are met
            if self._can_place_vnf_on_node(current_vnf, action, current_slice):
                node["hosted_vnfs"].append(current_vnf)
                part_path = nx.shortest_path(
                    self.topology.graph,
                    source=current_slice.path[-1],
                    target=action,
                    weight="latency",
                )
                current_slice.path = current_slice.path[:-1] + part_path
                node["cpu_usage"] += current_vnf.vcpu_usage

                if self.current_vnf_index == len(current_slice.vnf_list) - 1:
                    self._update_edges(current_slice)
                    if (
                        current_slice.vnf_list[self.current_vnf_index].vnf_type
                        != "Core"
                    ):
                        print(
                            f"WARNING! Last VNF is not Core type! something might be wrong"
                        )
                    done = self.current_slice_index == len(self.slices) - 1
                    if not done:
                        # print(f"NOT DONE!!!")
                        self.current_slice_index += 1
                        self.current_vnf_index = 0
                else:
                    self.current_vnf_index += 1
                    done = False
                # print(f"Before break. ACTION: {action}")
                reward = self._calculate_path_energy(current_slice)
                break
            else:
                attempts += 1
                action = (action + 1) % len(self.topology.graph.nodes)  # Try next node
                reward = -10  # Penalize each failed attempt to encourage exploration
                done = False
        # If maximum attempts are reached and VNF cannot be placed, terminate episode
        if attempts == max_attempts:
            print(
                f"Slice - {current_slice} - VNF {current_vnf.vnf_id} could not be placed. Terminating episode."
            )
            reward = -np.inf  # High penalty for unplaceable VNF
            done = True
        # print(f"DONE ACTION: {action}")
        return self._get_observation(), reward, done, {}

    def _update_edges(self, slice_obj):
        """Update bandwidth usage and availability after the slice is deployed"""
        edges = self.topology.graph.edges()
        for i in range(0, len(slice_obj.path) - 1):
            edge = edges[[slice_obj.path[i], slice_obj.path[i + 1]]]
            edge["link_usage"] += slice_obj.qos_requirement.bandwidth

    def _can_place_vnf_on_node(self, vnf, node_id, slice_obj):
        """Check if a VNF can be placed on a specific node considering resources and QoS constraints."""

        node = self.topology.graph.nodes[node_id]
        if (
            node["type"] != vnf.vnf_type
            or node["cpu_usage"] + vnf.vcpu_usage > node["cpu_limit"]
        ):
            return False
        if not self._check_latency_feasibility(
            node_id, slice_obj
        ) or not self._check_bandwidth_feasibility(node_id, slice_obj):
            return False
        return True

    def _check_latency_feasibility(self, node_id, slice_obj):
        """Check if latency requirements are feasible for the given node and slice."""
        # slice_obj.path[0] should always be the origin!!
        start_time = time.time()
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return True
        path = nx.shortest_path(
            self.topology.graph,
            source=slice_obj.path[-1],
            target=node_id,
            weight="latency",
        )
        cumulative_latency = nx.path_weight(self.topology.graph, path, weight="latency")
        elapsed_time = time.time() - start_time
        # print(f"_check_LATENCY_feasibility took {elapsed_time:.6f} seconds")
        return cumulative_latency <= slice_obj.qos_requirement.latency

    def _check_bandwidth_feasibility(self, node_id, slice_obj):
        """Check if bandwidth requirements are feasible for the given node and slice."""
        # slice_obj.path[0] should always be the origin!!
        start_time = time.time()
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return True
        path = nx.shortest_path(
            self.topology.graph,
            source=slice_obj.path[-1],
            target=node_id,
            weight="latency",
        )
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
        elapsed_time = time.time() - start_time
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
