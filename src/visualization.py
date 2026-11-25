import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Dict, Any
from src.network_builder import ThermalNetwork

def render_simulation_dashboard(network: ThermalNetwork, sim_settings: Dict[str, Any], time_steps: int, 
    history: npt.NDArray[np.float64], pump_history: npt.NDArray[np.float64]) -> None:

    print("Rendering simulation dashboard...")
    time_axis = np.linspace(0, sim_settings["total_time"], time_steps)
    spike_time = sim_settings["total_time"] / 2.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#9467bd', '#17becf', '#8c564b']
    linestyles = ['-', '--', '-.', ':']

    plot_idx = 0
    for i, node in enumerate(network.nodes):
        if node.__class__.__name__ == 'SiliconDie':
            ax1.plot(
                time_axis, history[:, i],
                label=f"{node.name.replace('_', ' ')} (°C)",
                color=colors[plot_idx % len(colors)],
                linestyle=linestyles[plot_idx % len(linestyles)],
                linewidth=2.0
            )
            plot_idx += 1

    ax1.axhline(sim_settings['target_temp'], color='black', linestyle='--', linewidth=1.2,
                label=f"PID Target ({sim_settings['target_temp']}°C)")
    ax1.axvline(spike_time, color='gray', linestyle=':', linewidth=1.2,
                label="Compute Load Spike (1.5x)")

    ax1.set_title("Active Server Rack: Closed-Loop PID Thermal Response", fontweight='bold', pad=10)
    ax1.set_ylabel("Junction Temperature (°C)", fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(time_axis, pump_history, color='#2ca02c', linewidth=2.0, label="Pump Flow Rate (kg/s)")
    ax2.axvline(spike_time, color='gray', linestyle=':', linewidth=1.2)
    ax2.set_xlabel("Time (s)", fontweight='bold')
    ax2.set_ylabel("Flow (kg/s)", fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.95)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("portfolio_thermal_response.png", dpi=300, bbox_inches="tight")
    plt.show()