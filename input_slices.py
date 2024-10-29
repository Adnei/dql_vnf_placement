from vnfs_and_slices import QoS, VNF, NetworkSlice

# List of NetworkSlice instances with QoS and VNFs
slices = [
    # Enhanced Mobile Broadband (eMBB) Slice
    NetworkSlice(
        slice_id=0,
        slice_type="eMBB",
        origin=0,  # Originating RAN node ID
        qos_requirement=QoS(
            qos_id=0, latency=100, edge_latency=None, bandwidth=50
        ),  # Max 20ms latency, 150 Mbps bandwidth
        vnf_list=[
            VNF(vnf_id=0, delay=0.2, vnf_type="RAN", vcpu_usage=2),  # RAN VNF
            VNF(vnf_id=1, delay=0.3, vnf_type="Edge", vcpu_usage=3),  # Edge VNF
            VNF(
                vnf_id=2, delay=0.2, vnf_type="Transport", vcpu_usage=2
            ),  # Transport VNF
            VNF(vnf_id=3, delay=0.4, vnf_type="Core", vcpu_usage=3),  # Core VNF
        ],
    ),
    # Ultra-Reliable Low-Latency Communication (URLLC) Slice
    NetworkSlice(
        slice_id=1,
        slice_type="URLLC",
        origin=1,  # Originating RAN node ID
        qos_requirement=QoS(
            qos_id=1, latency=10, edge_latency=2, bandwidth=100
        ),  # Max 5ms latency, stringent 2ms edge latency, 100 Mbps bandwidth
        vnf_list=[
            VNF(vnf_id=4, delay=0.1, vnf_type="RAN", vcpu_usage=1),  # RAN VNF
            VNF(
                vnf_id=5, delay=0.1, vnf_type="Edge", vcpu_usage=3
            ),  # Edge VNF (e.g., User Plane Function)
            VNF(
                vnf_id=6, delay=0.2, vnf_type="Transport", vcpu_usage=2
            ),  # Transport VNF
            VNF(vnf_id=7, delay=0.3, vnf_type="Core", vcpu_usage=2),  # Core VNF
        ],
    ),
    # Massive Machine-Type Communication (mMTC) Slice
    NetworkSlice(
        slice_id=2,
        slice_type="mMTC",
        origin=2,  # Originating RAN node ID
        qos_requirement=QoS(
            qos_id=2, latency=100, edge_latency=None, bandwidth=50
        ),  # Max 50ms latency, 50 Mbps bandwidth
        vnf_list=[
            VNF(vnf_id=8, delay=0.2, vnf_type="RAN", vcpu_usage=1),  # RAN VNF
            VNF(vnf_id=9, delay=0.3, vnf_type="Edge", vcpu_usage=2),  # Edge VNF
            VNF(
                vnf_id=10, delay=0.4, vnf_type="Transport", vcpu_usage=1
            ),  # Transport VNF
            VNF(vnf_id=11, delay=0.3, vnf_type="Core", vcpu_usage=2),  # Core VNF
        ],
    ),
    # Generic (OTHER) Slice
    NetworkSlice(
        slice_id=3,
        slice_type="OTHER",
        origin=3,  # Originating RAN node ID
        qos_requirement=QoS(
            qos_id=3, latency=60, edge_latency=None, bandwidth=75
        ),  # Max 30ms latency, 75 Mbps bandwidth
        vnf_list=[
            VNF(vnf_id=12, delay=0.2, vnf_type="RAN", vcpu_usage=2),  # RAN VNF
            VNF(vnf_id=13, delay=0.3, vnf_type="Edge", vcpu_usage=3),  # Edge VNF
            VNF(
                vnf_id=14, delay=0.3, vnf_type="Transport", vcpu_usage=2
            ),  # Transport VNF
            VNF(vnf_id=15, delay=0.4, vnf_type="Core", vcpu_usage=2),  # Core VNF
        ],
    ),
]
