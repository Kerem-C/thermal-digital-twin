import numpy as np
import matplotlib.pyplot as plt
from src.config_parser import build_network_from_yaml
from src.solver import ThermalSolver
from src.pid_controller import PIDController
from src.visualization import render_simulation_dashboard

def main() -> None:
    network, sim_settings = build_network_from_yaml("data/mixed_topology_rack.yaml")

    C, K_base, P, adv_indices = network.compile_static_matrices()
    solver = ThermalSolver(C, K_base, P, adv_indices)

    dt = sim_settings["dt"]
    time_steps = int(sim_settings["total_time"] / dt)
    T_current = np.array([node.initial_temp for node in network.nodes], dtype=np.float64)


    pid_track_index = next((i for i, node in enumerate(network.nodes) if node.__class__.__name__ == 'SiliconDie'), 0)

    pid = PIDController(kp=0.005, ki=0.0001, kd=0.001, 
                        setpoint=sim_settings["target_temp"], 
                        min_out=0.0, max_out=0.1)
    
    history = np.zeros((time_steps, len(network.nodes)), dtype=np.float64)
    pump_history = np.zeros(time_steps, dtype=np.float64)

    print(f"Executing Vectorized RK4 Integration over {time_steps} time steps...")

    for step in range(time_steps):
        history[step] = T_current

        gpu_temp = T_current[pid_track_index]
        m_dot = pid.compute(gpu_temp, dt)
        pump_history[step] = m_dot
        solver.update_advection(m_dot_total = m_dot)
        T_current = solver.step_rk4(T_current, dt)

    render_simulation_dashboard(network, sim_settings, time_steps, history, pump_history)

if __name__ == "__main__":
    main()