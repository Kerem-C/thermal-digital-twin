import numpy as np
import numpy.typing as npt
from typing import Tuple, List

class ThermalSolver:
    def __init__(self, 
                 C: npt.NDArray[np.float64],
                 K_base: npt.NDArray[np.float64],
                 P: npt.NDArray[np.float64],
                 advection_indices: List[Tuple[int, int, float, float]]) -> None:
        
        self.C = C
        self.K_base = K_base
        self.P = P
        self.advection_indices = advection_indices

        self.K_current = np.copy(K_base)

    def update_advection(self, m_dot_total: float) -> None:
       np.copyto(self.K_current, self.K_base)

       for idx_up, idx_down, flow_ratio, cp in self.advection_indices:
           G_adv = (m_dot_total * flow_ratio) * cp
           self.K_current[idx_up, idx_up] -= G_adv
           self.K_current[idx_down, idx_up] += G_adv

    def calculate_derivative(self, T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (np.dot(self.K_current, T) + self.P) / self.C

    def step_rk4(self, T_current: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        k1 = self.calculate_derivative(T_current)
        k2 = self.calculate_derivative(T_current + 0.5 * dt * k1)
        k3 = self.calculate_derivative(T_current + 0.5 * dt * k2)
        k4 = self.calculate_derivative(T_current + dt * k3)

        return T_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    

        