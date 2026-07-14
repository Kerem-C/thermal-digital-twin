import numpy as np
import scipy.sparse as sp
import pytest
from src.solver import ThermalSolver
from src.pid_controller import PIDController
from src.network_builder import ThermalNetwork
from src.nodes import CoolantFluidNode

def test_energy_conservation_steady_state():
    """
    If the power generated (P) is perfectly matched by heat leaving the system 
    through advection/conduction, the derivative (dT/dt) MUST equal 0.
    """
    C = np.array([100.0, 100.0])
    P = np.array([100.0, 0.0]) 
    K_base = sp.csr_matrix([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    
    solver = ThermalSolver(C, K_base, P, advection_indices=[])
    T_test = np.array([100.0, 0.0])
    
    T_next = solver.step_rk4(T_test, dt=1.0)
    
    np.testing.assert_almost_equal(T_next[0], T_test[0], decimal=2, err_msg="Energy is not conserved at steady state.")

def test_pid_boundary_enforcement():
    """
    Proves the PID controller respects physical hardware limits,
    even if the thermal error demands extreme responses.
    """
    pid = PIDController(kp=1.0, ki=0.1, kd=0.01, setpoint=55.0, min_out=0.0, max_out=0.1)
    
    # Simulate a massively overheated GPU (100C)
    max_flow = pid.compute(current_temp=100.0, dt=1.0)
    assert max_flow == 0.1, f"PID exceeded max_out. Expected 0.1, got {max_flow}"
    
    # Simulate an idle, freezing GPU (20C)
    min_flow = pid.compute(current_temp=20.0, dt=1.0)
    assert min_flow == 0.0, f"PID requested negative flow. Expected 0.0, got {min_flow}"

def test_mass_conservation():
    """
    Proves the network builder intercepts and rejects physically impossible flow routing
    at intermediate manifold nodes.
    """
    network = ThermalNetwork()
    
    # 0 is the Facility Supply (Source)
    # 1 is the Manifold (Intermediate)
    # 2 and 3 are the Branches (Sinks)
    network.add_node(CoolantFluidNode(node_id=0, name="Supply", mass=10.0))
    network.add_node(CoolantFluidNode(node_id=1, name="Manifold", mass=1.0))
    network.add_node(CoolantFluidNode(node_id=2, name="Branch_A", mass=1.0))
    network.add_node(CoolantFluidNode(node_id=3, name="Branch_B", mass=1.0))
    
    # Supply exactly 0.1 kg/s INTO the manifold
    network.add_advection_route(id_upstream=0, id_downstream=1, mass_flow_rate=0.1)
    
    # Intentionally violate mass conservation coming OUT of the manifold (0.1 in, but 0.15 out)
    network.add_advection_route(id_upstream=1, id_downstream=2, mass_flow_rate=0.1)
    network.add_advection_route(id_upstream=1, id_downstream=3, mass_flow_rate=0.05)
    
    # The network should flag Node 1 for violating the Continuity Equation
    with pytest.raises(ValueError, match="Mass conservation violated"):
        network.compile_static_matrices()