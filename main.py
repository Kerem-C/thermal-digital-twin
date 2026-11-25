import numpy as np
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

    # Track ALL SiliconDie nodes so the PID protects the hottest GPU in the rack
    die_indices = [i for i, node in enumerate(network.nodes) if node.__class__.__name__ == 'SiliconDie']

    pid = PIDController(kp=0.005, ki=0.0001, kd=0.001, 
                        setpoint=sim_settings["target_temp"], 
                        min_out=0.0, max_out=0.1)
    
    history = np.zeros((time_steps, len(network.nodes)), dtype=np.float64)
    pump_history = np.zeros(time_steps, dtype=np.float64)

    spike_step = time_steps // 2
    print(f"Executing Vectorized RK4 Integration over {time_steps} time steps...")

    for step in range(time_steps):
        # Inject a 1.5x compute load spike halfway through to demonstrate disturbance rejection
        if step == spike_step:
            solver.P[die_indices] *= 1.5

        history[step] = T_current

        # Feed the worst-case (hottest) GPU temperature to the PID controller
        hottest_gpu_temp = float(np.max(T_current[die_indices]))
        m_dot = pid.compute(hottest_gpu_temp, dt)
        pump_history[step] = m_dot

        solver.update_advection(m_dot_total=m_dot)
        T_current = solver.step_rk4(T_current, dt)

    render_simulation_dashboard(network, sim_settings, time_steps, history, pump_history)

if __name__ == "__main__":
    main()