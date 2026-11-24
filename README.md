# Dynamic Thermal Digital Twin

## Executive Summary
A discrete-time, hybrid Python and C++ thermodynamic simulation environment modeling liquid-cooled server rack topologies (including H100 GPU nodes and mixed-topology racks). The system translates declarative YAML hardware configurations into a continuous state-space matrix formulation, solved via a custom 4th-Order Runge-Kutta (RK4) integration engine with a `pybind11` C++ backend. To model active thermal management, fluid advection is dynamically regulated by a closed-loop PID controller targeting a 55°C GPU threshold. The architecture is decoupled and vectorized to achieve low-latency, deterministic execution across 200,000+ timesteps for stochastic heat load analysis and downstream Physics-Informed Neural Network (PINN) training.

---

## 1. System Architecture & Complexity Optimization
The core computational bottleneck of dynamic nodal networks is the recompilation of the Conductance Matrix ($K$) during variable fluid flow. This architecture solves this by decoupling static conduction from dynamic advection:

* **O(1) Matrix Updates:** Static thermal capacities ($C$) and solid-to-solid conduction ($K_{base}$) are compiled prior to execution. During the micro-timestep RK4 loop, PID-driven advection terms ($G_{adv}$) are injected exclusively into pre-mapped matrix indices using memory-safe pointer overwrites (`np.copyto`), shifting update complexity from $O(E)$ to $O(1)$.
* **C++ RK4 Backend (`pybind11`):** High-frequency numerical integration is offloaded to a compiled C++ engine (`src/cpp/rk4_engine.cpp`), enabling rapid evaluation of 200,000+ timesteps under stochastic compute loads.
* **Strict Typing & Validation:** The mathematical engine and state representations enforce strict typing compliance (`numpy.typing.NDArray`) and are verified against automated physics unit tests (`tests/test_physics.py`).
* **PINN Surrogate Pipeline:** Includes a dataset generation pipeline and PyTorch training scripts for Physics-Informed Neural Networks, along with research documentation analyzing PINN gradient pathology (`research/01_pinn_gradient_pathology.md`).

---

## 2. Project Structure
```text
thermal-digital-twin/
├── data/
│   ├── h100_server_node.yaml          # Single H100 node hardware definitions and thermal mass
│   ├── mixed_topology_rack.yaml       # Multi-node mixed server rack topology configuration
│   ├── thermal_training_data_v4.csv   # Generated stochastic thermal dataset
│   └── thermal_pinn.pth               # Trained PyTorch PINN model weights
├── research/
│   └── 01_pinn_gradient_pathology.md  # Technical notes on PINN gradient behavior
├── scripts/
│   ├── setup.py                       # Build script for the pybind11 C++ RK4 extension
│   ├── generate_dataset.py            # Stochastic Monte Carlo dataset generation
│   └── train_pinn.py                  # PyTorch PINN training pipeline
├── src/
│   ├── cpp/
│   │   └── rk4_engine.cpp             # Compiled C++ 4th-Order Runge-Kutta solver
│   ├── api.py                         # Programmatic interface for simulation execution
│   ├── config_parser.py               # YAML deserialization and dynamic system mapping
│   ├── network_builder.py             # Vectorized matrix compilation (static vs. dynamic)
│   ├── nodes.py                       # Physical component models (Silicon Die, Cold Plate, Fluid)
│   ├── pid_controller.py              # Dynamic regulation of pump mass flow rate
│   ├── solver.py                      # Python RK4 integration logic and O(1) matrix injectors
│   └── visualization.py               # Matplotlib dashboard and thermal response plotting
├── tests/
│   └── test_physics.py                # Automated physics and conservation unit tests
├── main.py                            # Single-scenario execution loop and dashboard generation
└── requirements.txt                   # Python dependencies
```

---

## 3. Mathematical Foundation
The core physics engine computes heat flow across the spatial grid using the continuous state-space derivative:

$$\frac{dT}{dt} = \frac{K \cdot T + P}{C}$$

**Dynamic Advection via PID:** Asymmetric advection conductance ($G_{adv}$) is calculated based on the fluid's specific heat capacity ($c_p$) and the instantaneous mass flow rate ($\dot{m}$) requested by the PID controller:

$$G_{adv} = \dot{m} \cdot c_p$$

---

## 4. Quickstart & Reproducibility

### Setup & Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/Kerem-C/thermal-digital-twin.git](https://github.com/Kerem-C/thermal-digital-twin.git)
cd thermal-digital-twin
pip install -r requirements.txt
```

Build the C++ RK4 extension (if compiling locally):
```bash
python scripts/setup.py build_ext --inplace
```

### Running the Simulation
Execute the default dynamic thermal simulation and generate the Matplotlib dashboard:
```bash
python main.py
```

### Testing & ML Pipeline
Run the physics validation suite:
```bash
pytest
```

Generate stochastic training data and train the Physics-Informed Neural Network:
```bash
python scripts/generate_dataset.py
python scripts/train_pinn.py
```
