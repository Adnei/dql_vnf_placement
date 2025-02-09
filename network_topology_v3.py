import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle


class NetworkTopologyGenerator:
    def __init__(self, from_file=None, n_nodes=50, avg_degree=4, rewire_prob=0.1):
        self.n_nodes = n_nodes
        self.avg_degree = avg_degree
        self.rewire_prob = rewire_prob
        self.node_types = {}
        if type(from_file) == str:
            self.graph = self.import_graph_from_pickle(from_file)
            self.node_types.update(
                {
                    node_id: "RAN"
                    for node_id, node_data in self.graph.nodes(data=True)
                    if node_data["type"] == "RAN"
                }
            )
            self.node_types.update(
                {
                    node_id: "Edge"
                    for node_id, node_data in self.graph.nodes(data=True)
                    if node_data["type"] == "Edge"
                }
            )
            self.node_types.update(
                {
                    node_id: "Transport"
                    for node_id, node_data in self.graph.nodes(data=True)
                    if node_data["type"] == "Transport"
                }
            )
            self.node_types.update(
                {
                    node_id: "Core"
                    for node_id, node_data in self.graph.nodes(data=True)
                    if node_data["type"] == "Core"
                }
            )
            print(f"Graph loaded from file: {from_file}")
            print(f"Graph: {self.graph}")
        else:
            self.graph = nx.connected_watts_strogatz_graph(
                n=n_nodes, k=avg_degree, p=rewire_prob
            )
            self.initialize_node_types()
            self.ensure_ran_edge_connections()
            self.add_redundant_paths()
            self.add_edge_interconnections()
            self.ensure_no_ran_transport_connections()
            self.ensure_no_ran_core_connections()
            self.ensure_no_core_edge_connections()
            self.add_edge_attributes()

    def initialize_node_types(self):
        """Assigns node types in a hierarchical manner (RAN, Edge, Transport, Core)."""

        node_types = ["RAN", "Edge", "Transport", "Core"]
        node_attributes = {
            "RAN": {
                "cpu_limit": random.randint(64, 128),
                "energy_base": random.uniform(50, 100),
                "energy_per_vcpu": random.uniform(3, 8),
            },
            "Edge": {
                "cpu_limit": random.randint(64, 128),
                "energy_base": random.uniform(100, 200),
                "energy_per_vcpu": random.uniform(6, 10),
            },
            "Transport": {
                "cpu_limit": random.randint(128, 256),
                "energy_base": random.uniform(200, 300),
                "energy_per_vcpu": random.uniform(8, 12),
            },
            "Core": {
                "cpu_limit": random.randint(256, 512),
                "energy_base": random.uniform(300, 400),
                "energy_per_vcpu": random.uniform(10, 15),
            },
        }

        ran_count = int(self.n_nodes * 0.4)
        edge_count = int(self.n_nodes * 0.3)
        transport_count = int(self.n_nodes * 0.2)
        core_count = self.n_nodes - (ran_count + edge_count + transport_count)

        self.node_types.update({node: "RAN" for node in range(ran_count)})
        self.node_types.update(
            {node: "Edge" for node in range(ran_count, ran_count + edge_count)}
        )
        self.node_types.update(
            {
                node: "Transport"
                for node in range(
                    ran_count + edge_count, ran_count + edge_count + transport_count
                )
            }
        )
        self.node_types.update(
            {
                node: "Core"
                for node in range(
                    ran_count + edge_count + transport_count, self.n_nodes
                )
            }
        )

        for node in self.graph.nodes():
            self.graph.nodes[node]["type"] = self.node_types[node]
            self.graph.nodes[node]["hosted_vnfs"] = []
            self.graph.nodes[node]["cpu_usage"] = 0
            self.graph.nodes[node].update(node_attributes[self.node_types[node]])

    def ensure_ran_edge_connections(self):
        """Ensures each RAN node is connected to at least one Edge node and avoids any RAN-RAN connections."""
        ran_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "RAN"
        ]
        edge_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Edge"
        ]

        # Remove any pre-existing RAN-RAN edges
        for ran_node in ran_nodes:
            neighbors = list(self.graph.neighbors(ran_node))
            for neighbor in neighbors:
                if self.node_types[neighbor] == "RAN":
                    self.graph.remove_edge(ran_node, neighbor)

        # Ensure each RAN node has at least one connection to an Edge node
        for ran_node in ran_nodes:
            if all(
                self.node_types[neighbor] != "Edge"
                for neighbor in self.graph.neighbors(ran_node)
            ):
                # Connect RAN node to a randomly chosen Edge node
                edge_node = random.choice(edge_nodes)
                self.graph.add_edge(ran_node, edge_node)

        # Optionally, add more connections between RAN and Edge nodes for redundancy
        for ran_node in ran_nodes:
            if (
                random.random() < 0.2
            ):  # 20% chance to add an extra connection to an Edge node
                edge_node = random.choice(edge_nodes)
                if not self.graph.has_edge(ran_node, edge_node):
                    self.graph.add_edge(ran_node, edge_node)

    def add_redundant_paths(self):
        """Adds redundant paths between Transport and Core nodes."""
        transport_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Transport"
        ]
        core_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Core"
        ]

        # Probabilistically add redundant paths between Transport and Core nodes
        for transport_node in transport_nodes:
            for core_node in core_nodes:
                if not self.graph.has_edge(transport_node, core_node):
                    if random.random() < 0.2:  # 20% chance to add an extra connection
                        self.graph.add_edge(transport_node, core_node)

    def add_edge_interconnections(self):
        """Adds random interconnections among Edge nodes for improved network resilience."""
        edge_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Edge"
        ]

        for i, node1 in enumerate(edge_nodes):
            for node2 in edge_nodes[i + 1 :]:
                if not self.graph.has_edge(node1, node2):
                    if (
                        random.random() < 0.1
                    ):  # 10% chance to add a connection between two Edge nodes
                        self.graph.add_edge(node1, node2)

    def ensure_no_ran_transport_connections(self):
        """Ensures no direct connections exist between RAN and Transport nodes."""
        ran_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "RAN"
        ]
        transport_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Transport"
        ]

        # Remove any RAN-Transport connections
        for ran_node in ran_nodes:
            for transport_node in transport_nodes:
                if self.graph.has_edge(ran_node, transport_node):
                    self.graph.remove_edge(ran_node, transport_node)

    def ensure_no_ran_core_connections(self):
        """Ensures no direct connections exist between RAN and Core nodes."""
        ran_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "RAN"
        ]
        core_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Core"
        ]

        # Remove any RAN-Core connections
        for ran_node in ran_nodes:
            for core_node in core_nodes:
                if self.graph.has_edge(ran_node, core_node):
                    self.graph.remove_edge(ran_node, core_node)

    def ensure_no_core_edge_connections(self):
        """Ensures no direct connections exist between Core and Edge nodes."""
        edge_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Edge"
        ]
        core_nodes = [
            node for node, n_type in self.node_types.items() if n_type == "Core"
        ]

        # Remove any Core-Edge connections
        for edge_node in edge_nodes:
            for core_node in core_nodes:
                if self.graph.has_edge(edge_node, core_node):
                    self.graph.remove_edge(edge_node, core_node)

    # @FIXME need more realistic values!
    def add_edge_attributes(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]["latency"] = random.uniform(1, 1.5)  # Latency in ms
            self.graph.edges[edge]["link_capacity"] = random.randint(
                15000, 50000
            )  # Link capacity in Mbps
            self.graph.edges[edge]["link_usage"] = 0  # Initialize link usage

    def draw(self, file_name="topology.pdf"):
        """Draws the network topology with colors indicating node types."""
        color_map = {
            "RAN": "red",
            "Edge": "blue",
            "Transport": "green",
            "Core": "purple",
        }
        colors = [color_map[self.node_types[node]] for node in self.graph.nodes()]

        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 12))
        nx.draw(
            self.graph,
            pos,
            node_color=colors,
            with_labels=True,
            node_size=500,
            font_size=8,
            font_weight="bold",
        )
        plt.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=type,
                )
                for type, color in color_map.items()
            ],
            loc="upper right",
        )
        plt.title("5G Network Topology")
        plt.savefig(file_name)
        plt.show()

    def get_graph(self):
        """Returns the generated networkx graph with the applied topology."""
        return self.graph

    def export_graph_to_pickle(self, filename="5G_Network_Topology.pickle"):
        pickle.dump(self.graph, open(filename, "wb"))

    def import_graph_from_pickle(self, filename="5G_Network_Topology.pickle"):
        return pickle.load(open(filename, "rb"))
