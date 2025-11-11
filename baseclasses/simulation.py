import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.signal import square
from typing import Union

from node import Node
from pipe import Pipe
from network import Network


class Simulation:

    def __init__(self, dt, total_time, net_id, T_ambt, temp_type = None, flow_type = None, file = None, no_cap = False):

        self.dt = dt
        self.total_time = total_time
        self.time = np.arange(0, total_time, dt) # time array
        self.num_steps = len(self.time) 
        self.T_ambt = T_ambt

        if flow_type == 'oscillation' or flow_type == 'square' or flow_type == 'constant':
            total_time_str = str(total_time)
        else:
            total_time_str = str(total_time - 1) 

        # Create simulation-specific subfolder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if file:
            sim_name = f"{file}_dt={dt}_Tambt={T_ambt}"
        else:
            sim_name = (
                f"network={net_id}_dt={dt}_total_time={total_time_str}_"
                f"Tin={temp_type}_mflow={flow_type}_Tambt={T_ambt}"
            )

        self.folder = os.path.join(base_dir, "figures", "simulation", sim_name)     
                   
        if no_cap:
            self.folder = self.folder + "_no_cap" 
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def simulate_network(self, 
                         network : Network, 
                         T_in : np.ndarray[Union[float]],
                         v_inflow : np.ndarray[Union[float]],
                         T_init_water : float,
                         T_init_pipe : float,
                         T_ambt: float,
                         plot_network = False,
                         plot_nodes_T = False,
                         plot_pipes_T = False,
                         plot_pipes_m_flow = False,
                         plot_nodes_dT = False,
                         plot_cap_influence = False,
                         plot_consumer_demand = False,
                         no_cap = False):
        """
        Simulate temperature dynamics for a network.
        
        Args:
        network: the network to be simulated
        T_in: array of inlet temperatures at the first node
        v_inflow: array of flow velocities at the first node   
        T_init: initial temperature in the network 
        T_ambt: ambient temperature
        """

        # T_in and v_inflow are saved in the network class
        network.initialize_network(self.dt, self.num_steps, v_inflow, T_in, T_init_water, T_init_pipe)

        for N in range(1,self.num_steps):
 
            network.set_T_and_flow_network(self.T_ambt, N, no_cap = no_cap)

        
        print('Simulation finished')

        # Plot outcome and save figure
        self.plot_network(network, plot = plot_network)
        self.plot_node_temperature_network(network, T_in, plot = plot_nodes_T)
        self.plot_pipe_temperature_network(network, T_in, plot = plot_pipes_T)
        self.plot_pipe_m_flow_network(network, v_inflow, plot = plot_pipes_m_flow)
        self.plot_node_difference_temperature_network(network, plot = plot_nodes_dT)
        self.plot_cap_influence(network, plot = plot_cap_influence)
        self.plot_consumer_demand(network, plot = plot_consumer_demand)
        self.save_data(network, T_in, v_inflow) 

        plt.show()  

    def plot_results_single_pipe_simulation(self, T_in, pipe, v_flow, decimal = 4):

        """
        Plot the results of the simulation
        time: time array for the simulation
        T_in: inlet temperature array
        pipe: pipe object with simulation results
        v_flow: flow velocity array
        decimal: number of decimals to round the temperature arrays to
        """
        plt.figure(figsize=(10, 6))
        plt.title("Water temperature")
        plt.ticklabel_format(style='plain', axis='y')  # Use plain formatting for y-axis

        plt.plot(self.time, T_in, label='Inlet Temperature')
        plt.plot(self.time, np.round(pipe.T_lossless, decimal), label = "Lossless temperature")
        plt.plot(self.time, np.round(pipe.T_cap, decimal), label='Temperature with pipe capacity')
        plt.plot(self.time, np.round(pipe.T, decimal), label = 'Real temperature')

        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(pipe.t_stay_array)
        plt.title('Average delay in the pipe')

        plt.figure()
        plt.plot(self.time, v_flow)
        plt.title('Flow velocity')

        plt.figure()
        plt.plot(self.time, pipe.m_flow)
        plt.title("Mass flow [m3/s]")
        plt.show()
    
    def plot_node_temperature_network(self, network: Network, T_in, plot = False):
        """
        Plot the temperature history for all nodes in the network
        
        Args:
            network: Network object containing the nodes to plot
        """
        fig_node_T = plt.figure(figsize=(10, 6))
        plt.title("Node Temperatures")
        
        for node_id, node in network.nodes.items():
            plt.plot(self.time, node.T, label=f'{node_id}')
      
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.folder + '/node_temperatures.png')

        fig_T_in = plt.figure()
        plt.plot(self.time, T_in)
        plt.title('Inlet temperature at first node')     
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)

        # Create directory if it doesn't exist
        plt.savefig(self.folder + '/inlet_temperature.png')

        if not plot:
            plt.close(fig_node_T)
            plt.close(fig_T_in)

    def plot_node_difference_temperature_network(self, network: Network, plot = False):
        
        fig = plt.figure()
        for _, pipe in network.pipes.items():

            node_to = pipe['to']
            node_from = pipe['from']

            dT = network.nodes[node_from].T - network.nodes[node_to].T 

            plt.plot(self.time, dT, label = 'dT' + node_from.split()[1] + node_to.split()[1])
        
        plt.title("Temperature differences between nodes")
        plt.ylabel("Temperature difference (°C)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)

        plt.savefig(self.folder + '/node_diff_temperatures.png')

        if not plot:
            plt.close(fig)
    
    def plot_pipe_temperature_network(self, network: Network, T_in, plot = False):
        """
        Plot the temperature history for all nodes in the network
        
        Args:
            network: Network object containing the nodes to plot
        """
        fig_pipe = plt.figure(figsize=(10, 6))
        plt.title('Temperature at outlet pipe')
        
        for pipe_id in network.pipes.keys():

            pipe = network.pipes[pipe_id]['pipe_instance']
            plt.plot(self.time, pipe.T, label=f'{pipe_id}, L = {pipe.L}')
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.folder + '/pipe_temperatures.png')

        if not plot:
            plt.close(fig_pipe)

    def plot_pipe_m_flow_network(self, network: Network, v_flow, plot = False):
        """
        Plot the temperature history for all nodes in the network
        
        Args:
            network: Network object containing the nodes to plot
        """
        fig_pipe_flow = plt.figure(figsize=(10, 6))
        plt.title("Pipe mass flows")
        
        for pipe_id in network.pipes.keys():

            pipe = network.pipes[pipe_id]['pipe_instance']
            # plt.figure()
            plt.plot(self.time, pipe.m_flow, label=f'{pipe_id}')
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/pipe_flows.png')


        pipe1 = next(iter(network.pipes.values()))['pipe_instance']


        fig_m_flow_in = plt.figure()
        plt.plot(v_flow * pipe1.inner_cs * pipe1.rho_water)
        plt.title('Mass in flow')
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Mass flow (kg/s)')
        plt.grid(True)

        plt.savefig(self.folder + '/inlet_mass_flow.png')

        if not plot:
            plt.close(fig_pipe_flow)
            plt.close(fig_m_flow_in)

    def plot_network(self, network: Network, plot = False):

        """
        Plot the network showing nodes as points and pipes as lines.
        Returns a matplotlib figure of the network.
        """
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot nodes
        for node_id, node in network.nodes.items():
            ax.scatter(node.x, node.y, node.z, c='red', marker='o')
            ax.text(node.x, node.y, node.z, node_id)

        # Plot pipes
        for pipe_id, pipe_info in network.pipes.items():
            from_node = network.nodes[pipe_info['from']]
            to_node = network.nodes[pipe_info['to']]

            # Pipe endpoints
            x_values = [from_node.x, to_node.x]
            y_values = [from_node.y, to_node.y]
            z_values = [from_node.z, to_node.z]

            # Plot the pipe line
            ax.plot(x_values, y_values, z_values, 'b-')

            # Compute midpoint of the pipe
            mid_x = (from_node.x + to_node.x) / 2
            mid_y = (from_node.y + to_node.y) / 2
            mid_z = (from_node.z + to_node.z) / 2

            # Add pipe number at the midpoint
            pipe_number = ' '.join(pipe_id.split()[1:])
            ax.text(mid_x, mid_y, mid_z, f'{pipe_number}', color='red', fontsize=14)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Network Layout')

        plt.savefig(self.folder + '/network.png')

        if not plot:
            plt.close(fig)

    def plot_cap_influence(self, network: Network, plot = False):
        """
        Plotting function to see the effect of the heat capacity plot
        """
        fig = plt.figure(figsize=(10, 6))
        plt.title("Pipe Capacity Influence")

        # Get the last pipe in the network
        last_pipe_id = list(network.pipes.keys())[-1]
        last_pipe = network.pipes[last_pipe_id]['pipe_instance']

        plt.plot(self.time, last_pipe.T_cap, label='T cap')
        plt.plot(self.time, last_pipe.T_lossless, label='T lossless')
        plt.plot(self.time, last_pipe.T, label='T real')

        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/cap_influence_last_pipe.png')

        if not plot:
            plt.close(fig)

    def plot_consumer_demand(self, network: Network, plot = False):
        """
        Plot the heat demand of all consumers in the network
        """
        fig = plt.figure(figsize=(10, 6))
        plt.title("Consumer Heat Demand")

        for hex_key in network.hexs.keys():
            hex = network.hexs[hex_key]['hex_instance']
            plt.plot(self.time, hex.consumer.Q_d, label=f'{hex.consumer.consumer_id}')

        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Heat Demand (W)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/consumer_heat_demand.png')

        if not plot:
            plt.close(fig)

    def save_data(self, network: Network, T_in, v_flow):
        """
        Save simulation data to CSV file.
        
        Args:
            network: Network object containing nodes and pipes
            T_in: Input temperature array
            v_flow: Input flow velocity array
        """
        # Initialize empty dictionary to store data
        data = {}
        
        # Store input data
        data['time'] = self.time
        data['T_in'] = T_in
        data['v_flow'] = v_flow
        
        # Store node temperatures
        for node_id, node in network.nodes.items():
            data[f'T_{node_id}'] = node.T
        
        # Store pipe mass flows and temperatures, and the temperature differences between nodes
        for pipe_id, pipe_info in network.pipes.items():
            pipe = pipe_info['pipe_instance']
            data[f'T {pipe_id}'] = pipe.T
            data[f'm_flow {pipe_id}'] = pipe.m_flow

        for pipe_id, pipe_info in network.pipes.items():
            node_from = pipe_info['from']
            node_to = pipe_info['to']
            data[f'dT {node_from.split()[1]}_{node_to.split()[1]}'] = network.nodes[node_from].T - network.nodes[node_to].T

        data['T_ambient'] = self.T_ambt

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.folder, 'simulation_data.csv'), index=False)

        # Store data for pipes #TODO: need to think of better way when using different pipes in one network.

        Network_data = {}

        Network_data['#nodes'] = len(network.nodes)
        Network_data['#pipes'] = len(network.pipes)

        pipe = next(iter(network.pipes.values()))['pipe_instance']
        Network_data['pipe_r_outer'] = pipe.r_outer
        Network_data['pipe_r_inner'] = pipe.r_inner
        Network_data['K'] = pipe.K
        Network_data["rho_pipe"] = pipe.rho_pipe
        Network_data["rho_insu"] = pipe.rho_insu 
        Network_data["cp_pipe"] = pipe.cp_pipe
        Network_data["cp_insu"] = pipe.cp_insu
        Network_data["insu_thickness"] = pipe.insu_thickness

        Index = [1]

        df_pipes = pd.DataFrame(Network_data, index = Index)
        df_pipes.to_csv(os.path.join(self.folder, 'pipe_data.csv'), index=False)

        # Saving the data corresponding to HEX and consumers
        HEX_data = {}
        
        for hex_key in network.hexs.keys():
            
            hex = network.hexs[hex_key]['hex_instance']

            HEX_data['Tc_in'] = hex.consumer.Tc_in
            HEX_data['Tc_out'] = hex.consumer.Tc_out
            HEX_data['mflow_consumer'] = hex.consumer.mflow

            HEX_data['Th_in'] = hex.pipes_in[f'pipe in {hex.node_id}'].T
            HEX_data['Th_out'] = hex.T
            HEX_data['mflow_hex'] = hex.pipes_in[f'pipe in {hex.node_id}'].m_flow
            HEX_data['Q_d'] = hex.consumer.Q_d

        df_hex = pd.DataFrame(HEX_data)
        df_hex.to_csv(os.path.join(self.folder, 'hex_consumer_data.csv'), index = False)
        

if __name__ == "__main__":
    pass 