import yaml
from src.nodes import SiliconDie, LiquidColdPlate, CoolantFluidNode
from src.network_builder import ThermalNetwork

def build_network_from_yaml(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    
    network = ThermalNetwork()

    node_type_mapping = {
        'SiliconDie': SiliconDie,
        'LiquidColdPlate': LiquidColdPlate,
        'CoolantFluidNode': CoolantFluidNode
    }

    for node_data in config['nodes']:
        NodeType = node_type_mapping[node_data['type']]

        kwargs = {
            'node_id': node_data['id'],
            'name': node_data['name'],
            'mass': node_data['mass'],
            'initial_temp': node_data['initial_temp']
        }

        if node_data['type'] == "SiliconDie":
            kwargs['power_gen'] = node_data.get('power_gen', 0.0)

        node = NodeType(**kwargs)
        network.add_node(node)

    for conn in config['connections']:
        network.add_connection(
            id_1=conn['id_1'],
            id_2=conn['id_2'],
            thermal_resistance=conn['resistance']
        )

    for flow in config['advection']:
        network.add_advection_route(
            id_upstream=flow['upstream_id'],
            id_downstream=flow['downstream_id'],
            mass_flow_rate=flow.get('mass_flow_rate', 0.0),
            specific_heat=flow.get('specific_heat', 4184.0)
        )    

    sim_settings = config['simulation']

    return network, sim_settings