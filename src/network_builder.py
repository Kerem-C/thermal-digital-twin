import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from typing import Tuple, List, Dict
from src.nodes import ThermalNode

class ThermalNetwork:

    def __init__(self) -> None:
        self.nodes: List[ThermalNode] = []
        self.node_indices: Dict[int, int] = {}
        self.conductance_map: Dict[Tuple[int, int], float] = {}
        self.advection_routes: List[Tuple[int, int, float, float]] = []

    def add_node(self, node: ThermalNode) -> None:
        index = len(self.nodes)
        self.nodes.append(node)
        self.node_indices[node.node_id] = index

    def add_connection(self, id_1: int, id_2: int, thermal_resistance: float) -> None:
        idx_1 = self.node_indices[id_1]
        idx_2 = self.node_indices[id_2]
        G = 1.0 / thermal_resistance
        self.conductance_map[(idx_1, idx_2)] = G
        self.conductance_map[(idx_2, idx_1)] = G

    def add_advection_route(self, id_upstream: int, id_downstream: int, mass_flow_rate: float, specific_heat: float = 4184.0) -> None:
        self.advection_routes.append((id_upstream, id_downstream, mass_flow_rate, specific_heat))

    def compile_static_matrices(self) -> Tuple[npt.NDArray[np.float64], sp.csr_matrix, npt.NDArray[np.float64], List[Tuple[int, int, float, float]]]:
        n = len(self.nodes)
        C = np.zeros(n, dtype=np.float64)
        P = np.zeros(n, dtype=np.float64)
        
        K_base: sp.lil_matrix = sp.lil_matrix((n, n), dtype=np.float64)

        for i, node in enumerate(self.nodes):
            C[i] = node.capacitance
            P[i] = node.power_gen
        
        for (i, j), G in self.conductance_map.items():
            K_base[i, j] += G
            K_base[i, i] -= G  

        m_dot_in: Dict[int, float] = {}
        m_dot_out: Dict[int, float] = {}
        total_system_flow = 0.0

        for up, down, m_dot, cp in self.advection_routes:
            m_dot_out[up] = m_dot_out.get(up, 0.0) + m_dot
            m_dot_in[down] = m_dot_in.get(down, 0.0) + m_dot
        
        for node_id, out_flow in m_dot_out.items():
            if node_id not in m_dot_in:
                total_system_flow += out_flow

        for node_id in self.node_indices.keys():
            if node_id in m_dot_in and node_id in m_dot_out:
                if not np.isclose(m_dot_in[node_id], m_dot_out[node_id], atol=1e-5):
                    raise ValueError(
                        f"CRITICAL FAULT: Mass conservation violated at Node ID {node_id}.\n"
                        f"Incoming flow: {m_dot_in[node_id]:.5f} kg/s | Outgoing flow: {m_dot_out[node_id]:.5f} kg/s"
                    )

        advection_indices = []
        for up, down, m_dot, cp in self.advection_routes:
            flow_ratio = (m_dot / total_system_flow) if total_system_flow > 0 else 0.0
            advection_indices.append((self.node_indices[up], self.node_indices[down], flow_ratio, cp))

        return C, K_base.tocsr(), P, advection_indices