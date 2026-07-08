import numpy as np
import matplotlib.pyplot as plt
from src.config_parser import build_network_from_yaml
from src.solver import ThermalSolver
from src.pid_controller import PIDController

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

    print("Rendering simulation dashboard...")
    time_axis = np.linspace(0, sim_settings["total_time"], time_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#17becf', '#bcbd22', '#e377c2', '#8c564b', '#7f7f7f']

    for i, node in enumerate(network.nodes):
        if node.__class__.__name__ == 'LiquidColdPlate':
            ax1.plot(time_axis, history[:, i], label=f"{node.name} (°C)",
                     color=colors[i % len(colors)], linewidth=2)
        
    ax1.axhline(sim_settings['target_temp'], color='black', linestyle='--',
                label=f"Target ({sim_settings['target_temp']}°C)")
    
    ax1.set_title("Active Server Rack: PID Closed-Loop Response", fontweight='bold')
    ax1.set_ylabel("Temperature (°C)", fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(time_axis, pump_history, color='#2ca02c', linewidth=2, label="Pump Flow Rate (kg/s)")

    ax2.set_xlabel("Time (s)", fontweight='bold')
    ax2.set_ylabel("Flow (kg/s)", fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()