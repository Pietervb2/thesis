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

    def __init__(self, dt, total_time, net_id, temp_type, flow_type, T_ambt, no_cap = False):

        self.dt = dt
        self.total_time = total_time
        self.time = np.arange(0, total_time, dt) # time array
        self.num_steps = len(self.time) 

        if flow_type == 'oscillating' or flow_type == 'square' or flow_type == 'constant':
            total_time_str = str(total_time)
        else:
            total_time_str = str(total_time - 1) 

        # Create simulation-specific subfolder
        self.folder = os.path.join('figures', 
                                   'network=' + net_id + "_" +
                                   'dt=' + str(dt) + '_' + 
                                   'total_time=' + total_time_str + "_" +
                                    'Tin=' + temp_type + "_" +
                                    'mflow=' + flow_type + '_' +
                                    'Tambt=' + str(T_ambt))
                   
        if no_cap:
            self.folder = self.folder + "_no_cap" 
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

  
    def simulate_pipe_temperature(self,
                                pipe: Pipe,
                                T_ambt : float,
                                T_inlet : np.ndarray[Union[float]],
                                v_flow : np.ndarray[Union[float]]):
        """
        Simulate temperature dynamics for a pipe section
                
        Args:
        T_inlet: array of inlet temperatures
        v_flow: array of flow velocities
        num_steps: number of simulation steps
        
        """

        # Initialize history
        pipe.bnode_init(self.dt, self.num_steps, v_flow, T_inlet, T_ambt) # NOTE: maybe use the minimum velocity as initial velocity
               
        # Run simulation with the extended arrays
        for N in range(self.num_steps):
            pipe.bnode_method(T_ambt, N)           
      
    def simulate_network(self, 
                         network : Network, 
                         T_in : np.ndarray[Union[float]],
                         v_inflow : np.ndarray[Union[float]],
                         T_ambt: float) -> None:
        """
        Simulate temperature dynamics for a network.
        
        Args:
        network: the network to be simulated
        dt: time step
        time: total time of simulation
        """

        network.initialize_network(self.dt, self.num_steps, v_inflow, T_in, T_ambt)

        for N in range(self.num_steps):
            print(f'N = {N}')
            network.set_T_and_flow_network(T_ambt, v_inflow[N], T_in[N], N)

    def plot_results_single_pipe_simulation(self, T_inlet, pipe, v_flow, decimal = 4):
        """
        Plot the results of the simulation
        time: time array for the simulation
        T_inlet: inlet temperature array
        pipe: pipe object with simulation results
        v_flow: flow velocity array
        decimal: number of decimals to round the temperature arrays to
        """
        plt.figure(figsize=(10, 6))
        plt.title("Water temperature")
        plt.ticklabel_format(style='plain', axis='y')  # Use plain formatting for y-axis

        plt.plot(self.time, T_inlet, label='Inlet Temperature')
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
    
    def plot_node_temperature_network(self, network: Network, T_inlet, plot = False):
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
        plt.plot(self.time, T_inlet)
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
    
    def plot_pipe_temperature_network(self, network: Network, T_inlet, plot = False):
        """
        Plot the temperature history for all nodes in the network
        
        Args:
            network: Network object containing the nodes to plot
        """
        fig_pipe = plt.figure(figsize=(10, 6))
        plt.title('Temperature at outlet pipe')
        
        for pipe_id in network.pipes.keys():

            pipe = network.pipes[pipe_id]['pipe_class']
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

            pipe = network.pipes[pipe_id]['pipe_class']
            # plt.figure()
            plt.plot(self.time, pipe.m_flow, label=f'{pipe_id}')
        plt.xlabel(f'Time (s), dt = {self.dt}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/pipe_flows.png')


        pipe1 = network.pipes['Pipe 1']['pipe_class']

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
            pipe_number = pipe_id.split()[1]
            ax.text(mid_x, mid_y, mid_z, f'{pipe_number}', color='red', fontsize=14)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Network Layout')

        plt.savefig(self.folder + '/network.png')

        if not plot:
            plt.close(fig)

    def save_data(self, network: Network, T_inlet, v_flow, T_ambt):
        """
        Save simulation data to CSV file.
        
        Args:
            network: Network object containing nodes and pipes
            T_inlet: Input temperature array
            v_flow: Input flow velocity array
        """
        # Initialize empty dictionary to store data
        data = {}
        
        # Store input data
        data['time'] = self.time
        data['T_inlet'] = T_inlet
        data['v_flow'] = v_flow
        
        # Store node temperatures
        for node_id, node in network.nodes.items():
            data[f'T_{node_id}'] = node.T
        
        # Store pipe mass flows and temperatures, and the temperature differences between nodes
        for pipe_id, pipe_info in network.pipes.items():
            pipe = pipe_info['pipe_class']
            data[f'T {pipe_id}'] = pipe.T
            data[f'm_flow {pipe_id}'] = pipe.m_flow

        for pipe_id, pipe_info in network.pipes.items():
            node_from = pipe_info['from']
            node_to = pipe_info['to']
            data[f'dT {node_from.split()[1]}_{node_to.split()[1]}'] = network.nodes[node_from].T - network.nodes[node_to].T

        data['T_ambient'] = T_ambt

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.folder, 'simulation_data.csv'), index=False)

        # Store data for pipes #TODO: need to think of better way when using different pipes in one network.

        data_pipes = {}
        pipe_1_instance = network.pipes['Pipe 1']['pipe_class']
        data_pipes['pipe_radius_outer'] = pipe_1_instance.radius_outer
        data_pipes['pipe_radius_inner'] = pipe_1_instance.radius_inner
        data_pipes['K'] = pipe_1_instance.K
        data_pipes["rho_pipe_mat"] = pipe_1_instance.rho_pipe_mat
        data_pipes["rho_insu"] = pipe_1_instance.rho_insu 
        data_pipes["cp_pipe_mat"] = pipe_1_instance.cp_pipe_mat
        data_pipes["cp_insu"] = pipe_1_instance.cp_insu
        data_pipes["insu_thickness"] = pipe_1_instance.insu_thickness

        Index = [1]

        df_pipes = pd.DataFrame(data_pipes, index = Index)
        df_pipes.to_csv(os.path.join(self.folder, 'pipe_data.csv'), index=False)
        


# Example test case
if __name__ == "__main__":

    # Pipe parameters
    pipe_radius_outer = 0.1 # m DUMMY
    pipe_radius_inner = 0.08 # m DUMMY
    K = 100 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
    pipe_data = [pipe_radius_outer, pipe_radius_inner, K]
      
    # Create network
    net = Network('test_id')
    net.add_node("Node 1", 0.0, 0.0, 0.0)
    net.add_node("Node 2", 1.0, 2.0, 30.0)
    net.add_pipe("Pipe 1", "Node 1", "Node 2", pipe_data)
   
    # Ambient temperature
    T_ambt = 15 # [C]

    # Time parameters
    dt = 1  # time step
    total_time = 300 # sec

    # Create simulation object
    sim = Simulation(dt, total_time)

    # Inlet temperature
    T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, sim.num_steps))   # Oscillating inlet temperature
    # T_inlet = np.ones(sim.num_steps) * 80                          # Constant
    # T_inlet = 80 + 1* square(2 * np.pi * sim.time / 20)                 # Square wave with a period of 20 steps
    
    # Flow velocity
    v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, sim.num_steps)) # Oscillating flow velocity
    # v_flow = np.ones(sim.num_steps) * 2                           # Constant
    # v_flow = 1.5 + 0.5 * square(2 * np.pi * sim.time / 50)        # Square wave flow velocity, 50 is the period

    pipe = (net.pipes['Pipe 1'])['pipe_class']
    sim.simulate_pipe_temperature(pipe, T_ambt, T_inlet, v_flow)
     
    # Plot results
    sim.plot_results_single_pipe_simulation(T_inlet, pipe, v_flow)

    # net.nodes['Node 2'].set_T() # For now it points to a whole array not to one single timestep. 
    # print(net.nodes['Node 2'].get_T())
    
    # Start interactive session
    # import code
    # code.interact(local=locals())