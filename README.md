# Dynamic Thermal Digital Twin

### Executive Summary
A discrete-time, Python-based thermodynamic simulation environment modeling a liquid-cooled server rack topology (e.g., H100 GPU architecture). The system translates declarative YAML hardware configurations into a continuous state-space matrix formulation, solved via a custom 4th-Order Runge-Kutta (RK4) integration engine. To model active thermal management, fluid advection is dynamically regulated by a PID controller targeting a 55°C GPU threshold. The architecture is strictly decoupled and vectorized with NumPy, designed to achieve low-latency execution for deterministic quantitative modeling and downstream machine learning dataset generation.

### 1. System Architecture & Complexity Optimization
The core computational bottleneck of dynamic nodal networks is the recompilation of the Conductance Matrix ($K$) during variable fluid flow. This architecture solves this by decoupling static conduction from dynamic advection.

* **$O(1)$ Matrix Updates:** Static thermal capacities ($C$) and solid-to-solid conduction ($K_{base}$) are compiled prior to execution. During the micro-timestep RK4 loop, the PID-driven advection terms ($G_{adv}$) are injected exclusively into pre-mapped matrix indices utilizing memory-safe pointer overwrites (`np.copyto`), shifting the update complexity from $O(E)$ to $O(1)$.
* **Strict Typing:** The mathematical engine and state representations enforce strict typing compliance, utilizing `numpy.typing.NDArray` for memory-safe state transitions.

### 2. File Structure
* `data/h100_server_node.yaml`: Declarative hardware definitions, thermal mass, and spatial routing.
* `src/`
    * `config_parser.py`: YAML deserialization and dynamic system mapping.
    * `nodes.py`: Object-oriented representations of physical components (Silicon Die, Cold Plate, Fluid Nodes).
    * `network_builder.py`: Vectorized matrix compilation isolating static baseline conductance from dynamic advection routes.
    * `solver.py`: Custom mathematical engine containing the numerical RK4 integration logic and $O(1)$ matrix injectors.
    * `pid_controller.py`: Dynamic regulation of pump mass flow rate ($\dot{m}$) based on thermal error margins.
* `main.py`: Single-scenario dynamic execution loop and Matplotlib dashboard generation.

### 3. Mathematical Foundation
The core physics engine computes heat flow across the spatial grid utilizing the continuous state-space derivative:

$$\frac{dT}{dt} = \frac{K \cdot T + P}{C}$$

**Dynamic Advection via PID:**
Asymmetric advection conductance ($G_{adv}$) is calculated based on the fluid's specific heat capacity ($c_p$) and the instantaneous mass flow rate ($\dot{m}$) requested by the PID controller:

$$G_{adv} = \dot{m} \cdot c_p$$

### 4. Quickstart Reproducibility
Clone the repository and execute the dynamic simulation:
```bash
git clone [https://github.com/Kerem-C/thermal-digital-twin.git](https://github.com/Kerem-C/thermal-digital-twin.git)
cd thermal-digital-twin
pip install -r requirements.txt
python main.py