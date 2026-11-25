import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from typing import Tuple, List

import rk4_backend

class ThermalSolver:
    def __init__(self, 
                 C: npt.NDArray[np.float64],
                 K_base: sp.csr_matrix,
                 P: npt.NDArray[np.float64],
                 advection_indices: List[Tuple[int, int, float, float]]) -> None:
        
        self.C = C
        self.P = P

        # Build the full CSR sparsity pattern once (static conduction + dynamic advection)
        K_pattern = K_base.tolil()
        for idx_up, idx_down, _, _ in advection_indices:
            K_pattern[idx_up, idx_up] = 1.0
            K_pattern[idx_down, idx_up] = 1.0
        K_csr = K_pattern.tocsr()

        self.K_indptr = K_csr.indptr.astype(np.int32)
        self.K_indices = K_csr.indices.astype(np.int32)
        self.K_base_data = np.zeros(len(K_csr.data), dtype=np.float64)

        # Populate static conduction values into the pre-allocated CSR data array
        K_base_coo = K_base.tocoo()
        for r, c, val in zip(K_base_coo.row, K_base_coo.col, K_base_coo.data):
            rs, re = self.K_indptr[r], self.K_indptr[r + 1]
            offset = int(np.where(self.K_indices[rs:re] == c)[0][0])
            self.K_base_data[rs + offset] = val

        self.K_data = self.K_base_data.copy()

        # Pre-map flat array indices for O(1) advection injection
        self.adv_flat_indices: List[Tuple[int, int, float]] = []
        for idx_up, idx_down, flow_ratio, cp in advection_indices:
            rs_u, re_u = self.K_indptr[idx_up], self.K_indptr[idx_up + 1]
            pos_up = rs_u + int(np.where(self.K_indices[rs_u:re_u] == idx_up)[0][0])

            rs_d, re_d = self.K_indptr[idx_down], self.K_indptr[idx_down + 1]
            pos_down = rs_d + int(np.where(self.K_indices[rs_d:re_d] == idx_up)[0][0])

            self.adv_flat_indices.append((pos_up, pos_down, flow_ratio * cp))

    def update_advection(self, m_dot_total: float) -> None:
        np.copyto(self.K_data, self.K_base_data)
        for pos_up, pos_down, factor in self.adv_flat_indices:
            G_adv = m_dot_total * factor
            self.K_data[pos_up] -= G_adv
            self.K_data[pos_down] += G_adv

    def step_rk4(self, T_current: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        return rk4_backend.step_rk4(
            T_current, dt, self.C, self.P, 
            self.K_data, self.K_indices, self.K_indptr
        )