import gymnasium as gym
import numpy as np
import networkx as nx


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

        # Place VNF if QoS and resource requirements are met
        if self._can_place_vnf_on_node(current_vnf, action, current_slice):
            node["hosted_vnfs"].append(current_vnf)
            current_slice.path.append(action)  # Add node to the slice path
            node["cpu_usage"] += current_vnf.vcpu_usage
            # @TODO -> Update edge with link usage += current_slice.qos_requirement.bandwidth

            # Check if Core VNF is placed, marking end of slice path
            # @TODO -> all vnfs are placed when current_vnf_index == len(self.slices.vnf_list) - 1
            if current_vnf.vnf_type == "Core":
                self.current_slice_index += 1
                self.current_vnf_index = 0
                done = self.current_slice_index >= len(self.slices)
            else:
                self.current_vnf_index += 1
                done = False

            reward = self._calculate_path_energy(current_slice)
        else:
            reward = -np.inf  # Penalize invalid placement
            done = False

        return self._get_observation(), reward, done, {}

    def _can_place_vnf_on_node(self, vnf, node_id, slice_obj):
        """Check if a VNF can be placed on a specific node considering resources and QoS constraints."""

        node = self.topology.graph.nodes[node_id]
        # if (
        #    len(slice_obj.path) == 0 and node != slice_obj.origin
        # ):  # We're looking for origin node
        #    return False
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
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return True
        path = nx.shortest_path(
            self.topology.graph,
            source=slice_obj.path[-1],
            target=node_id,
            weight="latency",
        )
        cumulative_latency = nx.path_weight(self.topology.graph, path, weight="latency")

        return cumulative_latency <= slice_obj.qos_requirement.latency

    def _check_bandwidth_feasibility(self, node_id, slice_obj):
        """Check if bandwidth requirements are feasible for the given node and slice."""
        # slice_obj.path[0] should always be the origin!!
        if node_id == slice_obj.path[0] and node_id == slice_obj.origin:
            return True
        path = nx.shortest_path(
            self.topology.graph,
            source=slice_obj.path[-1],
            target=node_id,
            weight="latency",
        )
        print(f"PATH: {path}")
        for node_idx in range(0, len(path) - 1):
            bandwidth_availability = (
                self.topology.graph.edges[path[node_idx], path[node_idx + 1]][
                    "link_capacity"
                ]
                - self.topology.graph.edges[path[node_idx], path[node_idx + 1]][
                    "link_usage"
                ]
            )

            print(f"Capacity: {bandwidth_availability}")
            if bandwidth_availability < slice_obj.qos_requirement.bandwidth:
                return False

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

        # Add energy costs along each link in the path
        for i in range(len(slice_obj.path) - 1):
            edge = self.topology.graph[slice_obj.path[i]][slice_obj.path[i + 1]]
            path_energy += edge["latency"] * edge["link_usage"]

        return -path_energy  # Negative reward for higher energy consumption

    def render(self, mode="human"):
        """Render the environment's current state."""
        print("Rendering environment state (extend for more detail)")
