import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Dict, Any
from src.network_builder import ThermalNetwork

def render_simulation_dashboard(network: ThermalNetwork, sim_settings: Dict[str, Any], time_steps: int, 
    history: npt.NDArray[np.float64], pump_history: npt.NDArray[np.float64]) -> None:

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