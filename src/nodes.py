class ThermalNode:
    def __init__(self, node_id: int, name: str, mass: float, specific_heat: float, initial_temp: float = 25.0, power_gen: float = 0.0) -> None:
        self.node_id: int = node_id
        self.name: str = name
        self.mass: float = mass                      # kg
        self.specific_heat: float = specific_heat    # J/(kg*K)
        self.temperature: float = initial_temp       # Celsius
        self.power_gen: float = power_gen            # Watts
        self.capacitance: float = self.mass * self.specific_heat # J/K
        self.initial_temp: float = initial_temp

class SiliconDie(ThermalNode):
    """Represents a CPU or GPU chip generating heat"""
    def __init__(self, node_id: int, name: str, mass: float, initial_temp: float = 25.0, power_gen: float = 0.0) -> None:
        super().__init__(node_id, name, mass, specific_heat=710.0, 
                         initial_temp=initial_temp, power_gen=power_gen)
        
class LiquidColdPlate(ThermalNode):
    def __init__(self, node_id: int, name: str, mass: float, initial_temp: float = 25.0) -> None:
        super().__init__(node_id, name, mass, specific_heat=385.0, 
                         initial_temp=initial_temp, power_gen=0.0)
        
class CoolantFluidNode(ThermalNode):
    def __init__(self, node_id: int, name: str, mass: float, initial_temp: float = 20.0) -> None:
        super().__init__(node_id, name, mass, specific_heat=4184.0, 
                         initial_temp=initial_temp, power_gen=0.0)