import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.tools as tls
import plotly.graph_objects as go

from pathlib import Path
from scipy.signal import square
from typing import Union

from node import Node
from pipe import Pipe
from network import Network


class Simulation:

    def __init__(self, dt, total_time, net_id, T_ambt, temp_type = None, file = None, no_cap = False):

        self.dt = dt
        self.total_time = total_time
        self.time = np.arange(0, total_time, dt) # time array
        self.num_steps = len(self.time) 
        self.T_ambt = T_ambt

        total_time_str = str(total_time)

        # Create simulation-specific subfolder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if file:
            sim_name = f"{file}_dt={dt}_Tambt={T_ambt}"
        else:
            sim_name = (
                f"network={net_id}_dt={dt}_total_time={total_time_str}_"
                f"Tin={temp_type}_Tambt={T_ambt}"
            )

        self.folder = os.path.join(base_dir, "figures", "simulation", sim_name)     
                   
        if no_cap:
            self.folder = self.folder + "_no_cap" 
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def simulate_network(self, 
                         network : Network, 
                         T_in : np.ndarray[Union[float]],
                         T_init_water : float,
                         T_init_pipe : float,
                         plot_network = False,
                         plot_nodes_T = False,
                         plot_pipes_T = False,
                         plot_pipes_mflow = False,
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
        network.initialize_network(self.dt, self.num_steps, T_in, T_init_water, T_init_pipe)

        for N in range(0,self.num_steps):
            
            network.set_mflow_network(N)
            network.set_T_network(self.T_ambt, N, no_cap = no_cap)
            # network.set_T_and_flow_network(self.T_ambt, N, no_cap = no_cap)

        
        print('Simulation finished')

        # Plot outcome and save figure
        self.plot_network(network, plot = plot_network)
        self.plot_node_temperature_network(network, T_in, plot = plot_nodes_T)
        self.plot_pipe_temperature_network(network, T_in, plot = plot_pipes_T)
        self.plot_pipe_mflow_network(network, plot = plot_pipes_mflow)
        self.plot_node_difference_temperature_network(network, plot = plot_nodes_dT)
        self.plot_cap_influence(network, plot = plot_cap_influence)
        self.plot_consumer_demand(network, plot = plot_consumer_demand)
        self.save_data(network, T_in) 

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

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        ax = plt.gca()
        ax.set_xlim(0, 24 * 3600)  # limits in seconds
        ticks_seconds = np.arange(0, 25, 4) * 3600
        ax.set_xticks(ticks_seconds)
        ax.set_xticklabels([f'{int(h)}' for h in np.arange(0, 25, 4)])
      
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='lower right')
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

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        ax = plt.gca()
        ax.set_xlim(0, 24 * 3600)  # limits in seconds
        ticks_seconds = np.arange(0, 25, 4) * 3600
        ax.set_xticks(ticks_seconds)
        ax.set_xticklabels([f'{int(h)}' for h in np.arange(0, 25, 4)])

        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.folder + '/pipe_temperatures.png')

        if not plot:
            plt.close(fig_pipe)

    def plot_pipe_mflow_network(self, network: Network, plot = False):
        """
        Plot the temperature history for all nodes in the network
        
        Args:
            network: Network object containing the nodes to plot
        """
        fig_pipe_flow = plt.figure(figsize=(10, 6))
        plt.title("Pipe mass flows")
        
        for pipe_id in network.pipes.keys():

            pipe = network.pipes[pipe_id]['pipe_instance']
            plt.plot(self.time, pipe.mflow, label=f'{pipe_id}')
        
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/pipe_flows.png')


        if not plot:
            plt.close(fig_pipe_flow)

    def plot_network_old(self, network: Network, plot = False):

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
            ax.text(mid_x, mid_y, mid_z, f'{pipe_number}', color='red', fontsize=10)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Network Layout')

        # Convert to Plotly figure and save as HTML
        plotly_fig = tls.mpl_to_plotly(fig)
        plotly_fig.write_html(self.folder + "/interactive_plot.html")

        # plt.savefig(self.folder + '/network.png')

        # if not plot:
        #     plt.close(fig)

    def plot_network(self, network, plot=False):
        fig = go.Figure()

        # Plot nodes
        for node_id, node in network.nodes.items():
            fig.add_trace(go.Scatter3d(
                x=[node.x], y=[node.y], z=[node.z],
                mode='markers+text',
                marker=dict(size=5, color='red'),
                text=[node_id],
                textposition='top center',
                name = node_id
            ))

        # Plot pipes
        for pipe_id, pipe_info in network.pipes.items():
            from_node = network.nodes[pipe_info['from']]
            to_node = network.nodes[pipe_info['to']]

            # Pipe line
            fig.add_trace(go.Scatter3d(
                x=[from_node.x, to_node.x],
                y=[from_node.y, to_node.y],
                z=[from_node.z, to_node.z],
                mode='lines',
                line=dict(color='blue', width=3),
                name=str(pipe_id)
            ))

            # Pipe label (midpoint)
            mid_x = (from_node.x + to_node.x) / 2
            mid_y = (from_node.y + to_node.y) / 2
            mid_z = (from_node.z + to_node.z) / 2

            fig.add_trace(go.Scatter3d(
                x=[mid_x], y=[mid_y], z=[mid_z],
                mode='text',
                text=[str(pipe_id.split(" ")[-1])],
                textfont=dict(color='red', size=10),
                showlegend=False
            ))

        # Layout
        fig.update_layout(
            title='Network Layout',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=900,
            height=900
        )

        # Save interactive HTML
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        html_path = Path(self.folder) / "interactive_plot.html"
        fig.write_html(html_path, auto_open = plot)

        # Optional: also export a static image
        # fig.write_image(Path(self.folder) / "network.png") TODO: requires orca installation

        # if plot:
        #     fig.show()

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

        if len(network.hexs) == 0:
            print("No HEXs in the network to plot consumer demand.")
            return
        
        fig = plt.figure(figsize=(10, 6))
        plt.title("Consumer Heat Demand vs Supply")

        for hex_key in network.hexs.keys():
            hex = network.hexs[hex_key]
            plt.plot(self.time, hex.consumer.Q_d, label=f'Heat demand of C{hex.consumer.consumer_id.split(" ")[1]}')
            plt.plot(self.time, hex.consumer.Q_supply, label=f'Heat supplied to C{hex.consumer.consumer_id.split(" ")[1]}', linestyle='--')

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        ax = plt.gca()
        ax.set_xlim(0, 24 * 3600)  # limits in seconds
        ticks_seconds = np.arange(0, 25, 4) * 3600
        ax.set_xticks(ticks_seconds)
        ax.set_xticklabels([f'{int(h)}' for h in np.arange(0, 25, 4)])

        plt.xlabel(f'Time (hours), dt = {self.dt}')
        plt.ylabel('Heat Demand (W)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/HEAT.png')

        if not plot:
            plt.close(fig)

    def save_data(self, network: Network, T_in):
        """
        Save simulation data to CSV file.
        
        Args:
            network: Network object containing nodes and pipes
            T_in: Input temperature array
        """

        sim_data_folder = os.path.join(self.folder, 'simulation_data')
        if not os.path.exists(sim_data_folder):
            os.makedirs(sim_data_folder)
        
        # Initialize empty dictionaries to store data
        node_data = {}
        node_dT_data = {}

        pipe_T_data = {}
        pipe_mflow_data = {}
        pipes_dp_data = {}
                       
        # Store time in all dicts
        node_data['time'] = self.time
        node_dT_data['time'] = self.time
        pipe_T_data['time'] = self.time
        pipe_mflow_data['time'] = self.time

        # Inlet temperature
        node_data['T_in'] = T_in

        for node_id, node in network.nodes.items():
            node_data[f'{node_id}'] = np.round(node.T,3)

        node_data['T_ambient'] = self.T_ambt
        
        # Store pipe mass flows and temperatures, and the temperature differences between nodes

        for pipe_id, pipe_info in network.pipes.items():
            pipe = pipe_info['pipe_instance']
            pipe_T_data[f'{pipe_id}'] = np.round(pipe.T,3)
            pipe_mflow_data[f'{pipe_id}'] = np.round(pipe.mflow,5)
            pipes_dp_data[f'{pipe_id}'] = np.round(pipe.dp_friction,3)
        
        for pipe_id, pipe_info in network.pipes.items():
            node_from = pipe_info['from']
            node_to = pipe_info['to']
            node_dT_data[f'dT {node_from.split()[1]}_{node_to.split()[1]}'] = np.round(network.nodes[node_from].T - network.nodes[node_to].T,3)

        # Save simulation data
        df_node = pd.DataFrame(node_data)
        df_node_dT = pd.DataFrame(node_dT_data)
        df_pipe_T = pd.DataFrame(pipe_T_data)
        df_pipe_mflow = pd.DataFrame(pipe_mflow_data)
        df_pipe_dp = pd.DataFrame(pipes_dp_data)

        df_node.to_csv(os.path.join(sim_data_folder, 'Node_temp.csv'), index=False)
        df_node_dT.to_csv(os.path.join(sim_data_folder,'Node_dT.csv'),index = False)
        df_pipe_T.to_csv(os.path.join(sim_data_folder,'Pipe_temp.csv'),index=False)
        df_pipe_mflow.to_csv(os.path.join(sim_data_folder,'Pipe_mflow.csv'), index =  False)
        df_pipe_dp.to_csv(os.path.join(sim_data_folder,'Pipe_dp_friction.csv'), index = False)
        
        # Network data incombination with the pipe properites
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
        hex_dp_data = {}

        hex_folder = os.path.join(self.folder, 'hex_consumer_data')
        if not os.path.exists(hex_folder):
            os.makedirs(hex_folder)

        for hex_key in network.hexs.keys():
            
            hex = network.hexs[hex_key]

            HEX_data['Tc_in'] = hex.consumer.Tc_in
            HEX_data['Th_in'] = hex.pipes_in[f'Pipe {hex_key.split()[-1]}.1'].T
            HEX_data['Tc_out'] = hex.consumer.Tc_out
            HEX_data['Th_out'] = hex.T
            HEX_data['mflow_prim'] = hex.pipes_in[f'Pipe {hex_key.split()[-1]}.1'].mflow
            HEX_data['mflow_sec'] = hex.consumer.mflow           
            HEX_data['Q_d'] = hex.consumer.Q_d
            HEX_data['Q_supply'] = hex.consumer.Q_supply

            hex_dp_data[f'{hex_key}'] = hex.pressure_drop() * hex.pipes_in[f'Pipe {hex_key.split()[-1]}.1'].mflow**2

            df_hex = pd.DataFrame(HEX_data)
            df_hex_dp = pd.DataFrame(hex_dp_data)

            df_hex_dp.to_csv(os.path.join(self.folder,'hex_consumer_data',f'{hex_key}_dp.csv'), index = False)
            df_hex.to_csv(os.path.join(self.folder,'hex_consumer_data',f'{hex_key}.csv'), index = False)
        

if __name__ == "__main__":
    pass 