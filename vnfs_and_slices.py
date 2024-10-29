import random


class QoS:
    def __init__(self, qos_id, latency, edge_latency, bandwidth):
        self.qos_id = qos_id
        self.latency = latency
        self.edge_latency = edge_latency
        self.bandwidth = bandwidth

    def __str__(self):
        return f"<QoS qos_id:{self.qos_id}, latency:{self.latency}, bandwidth:{self.bandwidth}, edge_latency:{self.edge_latency}>"


class VNF:
    def __init__(self, vnf_id, vnf_type, delay, vcpu_usage):
        self.vnf_id = vnf_id
        self.vnf_type = vnf_type
        self.delay = delay
        self.vcpu_usage = vcpu_usage

    def __str__(self):
        return f"<VNF vnf_id:{self.vnf_id}, vcpu_usage:{self.vcpu_usage}, delay:{self.delay}>"


class NetworkSlice:
    def __init__(self, slice_id, slice_type, qos_requirement, origin, vnf_list):
        self.slice_id = slice_id
        self.slice_type = slice_type
        self.qos_requirement = qos_requirement
        self.origin = origin
        self.vnf_list = vnf_list
        self.core_node = None
        self.path = []  # Updated to store full path with edges

    def add_to_path(self, node, edge=None):
        self.path.append((node, edge))  # Save node and edge in path

    def __str__(self):
        return f"<NetworkSlice slice_id:{self.slice_id}, qos_requirement:{self.qos_requirement}, origin:{self.origin}, core_node:{self.core_node}, path:{self.path}, vnf_list:{self.vnf_list}>"
