class QoS:
    def __init__(self, qos_id, latency, bandwidth):
        self.qos_id = qos_id
        self.latency = latency
        self.bandwidth = bandwidth


class VNF:
    def __init__(self, vnf_id, delay, vcpu_usage, network_usage):
        self.vnf_id = vnf_id
        self.delay = delay
        self.vcpu_usage = vcpu_usage
        self.network_usage = network_usage


class NetworkSlice:
    def __init__(
        self, slice_id, slice_type, qos_requirement, origin, core_node, vnf_list
    ):
        self.slice_id = slice_id
        self.slice_type = slice_type
        self.qos_requirement = qos_requirement
        self.origin = origin
        self.core_node = core_node
        self.vnf_list = vnf_list
        self.path = None
