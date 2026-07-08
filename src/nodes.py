class ThermalNode:
    def __init__(self, node_id, name, mass, specific_heat, initial_temp=25.0, power_gen=0.0):
        self.node_id = node_id
        self.name = name
        self.mass = mass                    # kg
        self.specific_heat = specific_heat  # J/(kg*K)
        self.temperature = initial_temp     # Celcius
        self.power_gen = power_gen          # Watts
        self.capacitance = self.mass * self.specific_heat # J/K
        self.initial_temp = initial_temp

class SiliconDie(ThermalNode):
    """Represents a CPU or GPU chip generating heat"""
    def __init__(self, node_id, name, mass, initial_temp=25.0, power_gen=0.0):
        super().__init__(node_id, name, mass, specific_heat=710.0, 
                         initial_temp=initial_temp, power_gen=power_gen)
        
class LiquidColdPlate(ThermalNode):
    def __init__(self, node_id, name, mass, initial_temp=25.0):
        super().__init__(node_id, name, mass, specific_heat=385.0, 
                         initial_temp=initial_temp, power_gen=0.0)
        
class CoolantFluidNode(ThermalNode):
    def __init__(self, node_id, name, mass, initial_temp=20.0):
        super().__init__(node_id, name, mass, specific_heat=4184.0, 
                         initial_temp=initial_temp, power_gen=0.0)