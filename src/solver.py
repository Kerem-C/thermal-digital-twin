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
        self.K_base = K_base
        self.P = P
        self.advection_indices = advection_indices

        self.K_current: sp.csr_matrix = self.K_base.copy()

    def update_advection(self, m_dot_total: float) -> None:
        self.K_current = self.K_base.copy()

        K_mod: sp.lil_matrix = self.K_current.tolil()

        for idx_up, idx_down, flow_ratio, cp in self.advection_indices:
           G_adv = (m_dot_total * flow_ratio) * cp
           K_mod[idx_up, idx_up] -= G_adv
           K_mod[idx_down, idx_up] += G_adv
        
        self.K_current = K_mod.tocsr()

    def step_rk4(self, T_current: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        K_data = self.K_current.data.astype(np.float64)
        K_indices = self.K_current.indices.astype(np.int32)
        K_indptr = self.K_current.indptr.astype(np.int32)
        
        T_next = rk4_backend.step_rk4(
            T_current, dt, self.C, self.P, 
            K_data, K_indices, K_indptr
        )
        
        return T_next

    

        