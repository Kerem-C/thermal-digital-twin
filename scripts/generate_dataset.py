import numpy as np
import pandas as pd
from typing import Dict
import time

from src.config_parser import build_network_from_yaml
from src.solver import ThermalSolver
from src.pid_controller import PIDController

def run_monte_carlo(num_simulations: int = 500) -> None:
    print(f"Initializing Monte Carlo Engine for {num_simulations} stochastic scenarios...")

    network, sim_settings = build_network_from_yaml("data/mixed_topology_rack.yaml")
    dt = sim_settings["dt"]
    time_steps = int(sim_settings["total_time"] / dt)
    target_temp = sim_settings["target_temp"]

    silicon_indices = [i for i, node in enumerate(network.nodes) if node.__class__.__name__ == 'SiliconDie']
    num_gpus = len(silicon_indices)

    dataset: Dict[str, np.ndarray] = {
        "p_gpu_1": np.zeros(num_simulations, dtype=np.float64),
        "p_gpu_2": np.zeros(num_simulations, dtype=np.float64),
        "p_gpu_3": np.zeros(num_simulations, dtype=np.float64),
        "p_gpu_4": np.zeros(num_simulations, dtype=np.float64),
        "m_dot_steady_kg_s": np.zeros(num_simulations, dtype=np.float64),
        "t_return_manifold": np.zeros(num_simulations, dtype=np.float64)
    }

    p_gpu_samples = np.random.uniform(300.0, 1000.0, (num_simulations, num_gpus))

    start_time = time.time()

    for i in range(num_simulations):
        for idx, sil_idx in enumerate(silicon_indices):
            network.nodes[sil_idx].power_gen = p_gpu_samples[i, idx]

        C, K_base, P, adv_indices = network.compile_static_matrices()
        solver = ThermalSolver(C, K_base, P, adv_indices)

        T_current = np.array([node.initial_temp for node in network.nodes], dtype=np.float64)

        pid = PIDController(kp=0.005, ki=0.0001, kd=0.001,
                            setpoint=target_temp, min_out=0.0, max_out=0.1)
        
        steady_m_dot = 0.0

        for step in range(time_steps):
            max_gpu_temp = np.max(T_current[silicon_indices])
            m_dot = pid.compute(max_gpu_temp, dt)

            solver.update_advection(m_dot_total=m_dot)
            T_current = solver.step_rk4(T_current, dt)

            if step == time_steps - 1:
                steady_m_dot = m_dot

        dataset["p_gpu_1"][i] = p_gpu_samples[i, 0]
        dataset["p_gpu_2"][i] = p_gpu_samples[i, 1]
        dataset["p_gpu_3"][i] = p_gpu_samples[i, 2]
        dataset["p_gpu_4"][i] = p_gpu_samples[i, 3]
        dataset["m_dot_steady_kg_s"][i] = steady_m_dot

        return_node_idx = network.node_indices.get(999, -1)
        if return_node_idx != -1:
            dataset["t_return_manifold"][i] = T_current[return_node_idx]

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_simulations} scenarios...")        

    execution_time = time.time() - start_time
    print(f"Simulation complete in {execution_time:.2f} seconds.")

    df = pd.DataFrame(dataset)
    output_path = "data/thermal_training_data_v4.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully exported to {output_path} with {len(df)} records.")

if __name__ == "__main__":
    run_monte_carlo()