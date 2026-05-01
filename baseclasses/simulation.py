from .network import Network

from pathlib import Path
from scipy.signal import square
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.graph_objects as go

class Simulation:

    def __init__(self, profile, run_type, dt, total_time, T_ambt, 
                 test_name = None,
                 net_id = None, 
                 file = None, 
                 no_cap = False, 
                 n_init_points = None,
                 n_iter = None):

        self.dt = dt
        self.total_time = total_time
        self.time = np.arange(0, total_time, dt) # time array
        self.num_steps = len(self.time) 
        self.T_ambt = T_ambt

        total_time_str = str(total_time)

        # Create simulation-specific subfolder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
           
        if run_type != 'optimization':
            if run_type == 'benchmark':
                self.folder = os.path.join(base_dir, "figures", "benchmark", f"Benchmark_{profile}_dt={dt}")

            elif run_type == 'save_optimization':
                self.folder = os.path.join(base_dir, "figures", "optimization_PI", f"{profile}_dt={dt}_init_points={n_init_points}_n_iter={n_iter}")
            
            elif run_type == 'test': 
                if file:
                    sim_name = f"{file}_dt={dt}_Tambt={T_ambt}"
                else:
                    
                    if test_name is not None:
                        sim_name = f"{test_name}_dt={dt}_Tambt={T_ambt}"
                    else:
                        sim_name = (
                            f"{net_id}_dt={dt}_total_time={total_time_str}_"
                            f"Tambt={T_ambt}"
                    )

                self.folder = os.path.join(base_dir, "figures", "simulation", sim_name)     
                    
                if no_cap:
                    self.folder = self.folder + "_no_cap"       

            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

    def simulate_network(self, 
                         run_type, 
                         network : Network, 
                         T_init_water : float,
                         T_init_pipe : float,
                         theta_1 = None,
                         theta_2 = None,
                         theta_3 = None,
                         theta_4 = None,
                         T_in = None,
                         plot_network = False,
                         plot_sup_ret = False,
                         plot_pipes_T = False,
                         plot_overflow = False,
                         plot_consumer_demand = False,
                         plot_h_valves = False,
                         no_cap = False,
                         plot_pipes_mflow = False,
                         plot_nodes_dT = False,
                         plot_cap_influence = False,
                         plot_nodes_T = False):
        """
        Simulate temperature dynamics for a network.
        
        Args:
        network: the network to be simulated
        T_in: array of inlet temperatures at the first node
        v_inflow: array of flow velocities at the first node   
        T_init: initial temperature in the network 
        T_ambt: ambient temperature
        """

        if T_in is not None:
            network.initialize_network(self.dt, self.num_steps, T_in[0], T_init_water, T_init_pipe)
        else:
            network.initialize_network(self.dt, self.num_steps, theta_1, T_init_water, T_init_pipe) # need to create Q_d, theta_1 stated as it will always start with this low temp. Afterwards T_in for rest of simulation will be determined.
            T_in = self.supply_temperature_BO(theta_1, theta_2, theta_3, theta_4, network, self.total_time, self.dt)

        for N in range(0,self.num_steps):

            network.set_mflow_network(N)
            network.set_T_network(self.T_ambt, N, T_in[N], no_cap = no_cap)
            
        if run_type != 'optimization':
            print('Simulation finished')

            # Plot outcome and save figure
            self.plot_network(network, plot = plot_network)
            self.plot_return_pipe_temperature_network(network, T_in, plot = plot_pipes_T)
            self.plot_supply_return_temperature_flow(network, plot = plot_sup_ret)
            self.plot_overflow(network, plot = plot_overflow)
            self.plot_consumer_demand(network, plot = plot_consumer_demand)
            self.plot_h_valves(network, plot = plot_h_valves)
            self.save_data(network, T_in) 

            # self.plot_node_temperature_network(network, plot = plot_nodes_T)
            # self.plot_pipe_mflow_network(network, plot = plot_pipes_mflow)
            # self.plot_node_difference_temperature_network(network, plot = plot_nodes_dT)
            # self.plot_cap_influence(network, plot = plot_cap_influence)
            plt.show()  
      
    def supply_temperature_BO(self, theta_1, theta_2, theta_3, theta_4, net : Network, total_time, dt):
        """
        Calculate the total heat demand of all consumers in the network and determine the supply temperature.  
        """
        time = np.arange(0, total_time, dt)
        num_steps = len(time)
        total_heat_demand = np.zeros(num_steps)
        T_supply = np.zeros(num_steps)
        
        # Iterate through all heat exchangers and sum consumer demands
        for hex_obj in net.hexs.values():
            consumer = hex_obj.consumer
            total_heat_demand += consumer.Q_d
        
        # Look 3 minutes into time. 
        look_ahead = int(3 * 60 / dt)  # 3 minutes in steps
        for i in range(num_steps):

            if i + look_ahead >= num_steps:
                k = i
            else:
                k = i + look_ahead

            # if total_heat_demand[k] < theta_3:
            #     T_supply[i] = theta_1
            # elif total_heat_demand[k] >= theta_3 + theta_4:
            #     T_supply[i] = theta_2
            # else:
            #     T_supply[i] = theta_1 + (total_heat_demand[k] - theta_3) * (theta_2 - theta_1) / theta_4

            T_supply[i] = theta_1 + theta_2 * np.tanh(theta_3 * (total_heat_demand[k] - theta_4)) # Alternative formulation with tanh function

        return T_supply 
    
    def plot_node_temperature_network(self, network: Network, plot = False):
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
        format_plot_time_axis(num_hours=24)
      
        plt.xlabel(f'Time (h), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.savefig(self.folder + '/node_temperatures.png')

        if not plot:
            plt.close(fig_node_T)

    def plot_supply_return_temperature_flow(self, network: Network, plot = False):
        
        fig_T_in = plt.figure()       
        plt.plot(self.time, network.nodes['Node 1.1'].T, label = 'Supply temperature')
        plt.plot(self.time, network.nodes['Node 1.6'].T, label = 'Return temperature')
        plt.title('Supply and Return Temperature')     
        plt.xlabel(f'Time (h), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        # Create directory if it doesn't exist
        plt.savefig(self.folder + '/supply_return_temperature.png')

        fig_mflow = plt.figure()

        plt.plot(self.time, network.pipes['Pipe 1.1']['pipe_instance'].mflow, label = 'Supply')
        plt.plot(self.time, network.pipes['Pipe 1.6']['pipe_instance'].mflow, label = 'Return')
        plt.title('Supply and Return Mass Flow Rate')
        plt.xlabel(f'Time (h), dt = {self.dt}')
        plt.ylabel('Mass Flow Rate (kg/s)')
        plt.legend()
        plt.grid(True)

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        plt.savefig(self.folder + '/supply_return_mflow.png')

        if not plot:
            plt.close(fig_T_in)
            plt.close(fig_mflow)

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
    
    def plot_return_pipe_temperature_network(self, network: Network, T_in, plot = False):
        """
        Plot the temperature history for all nodes in the network
        
        Args:
            network: Network object containing the nodes to plot
        """
        fig_pipe = plt.figure(figsize=(10, 6))
        plt.title('Temperature at outlet pipe')
        plot_pipe = 1
        for pipe_id in network.pipes.keys():

            if '.6' in pipe_id:  # Return riser
                plot_pipe = plot_pipe*-1
                pipe = network.pipes[pipe_id]['pipe_instance']
                if plot_pipe == 1:
                    plt.plot(self.time, pipe.T, label=f'{pipe_id}, L = {pipe.L}')

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        plt.xlabel(f'Time (h), dt = {self.dt}')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature at end of return pipes')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.folder + '/return_pipe_temperatures.png')

        if not plot:
            plt.close(fig_pipe)
        
    def plot_overflow(self, network: Network, plot = False): 

        fig_overflow_T = plt.figure(figsize=(10, 6)) 
        plt.title("Overflow Temperature") 
        for valve in network.valves.values(): 
            if 'Overflow' in valve.valve_id: 
                temp = valve.node.T
                valve_h = valve.h

        plt.plot(self.time, temp) 
        plt.plot(self.time, np.ones_like(temp)*55, linestyle = 'dashed', color = 'grey', linewidth=0.8)

        num_hours = int(self.total_time / 3600)
                      
        format_plot_time_axis(num_hours=num_hours)

        plt.xlabel(f'Time (hours), dt = {self.dt}') 
        plt.ylabel('Temperature (°C)') 
        plt.grid(True)
        
        # Add custom tick label for setpoint without removing automatic ticks
        ax = plt.gca()
        ax.axhline(55, linestyle='dotted', color = 'grey', linewidth=1, label=r'$T_{set}$')
        
        plt.savefig(self.folder + '/overflow_temperature.png')

        fig_overflow_mflow = plt.figure()

        plt.plot(self.time, network.pipes['Overflow 1']['pipe_instance'].mflow)
        format_plot_time_axis(num_hours=24)

        plt.title("Overflow Mass Flow Rate")
        plt.xlabel(f'Time (h), dt = {self.dt}') 
        plt.ylabel('Mass Flow Rate (kg/s)') 
        plt.grid(True)
        plt.savefig(self.folder + '/overflow_mflow.png')

        fig_overflow_h = plt.figure()

        # Plot overflow valve displacement for each valve
        for valve in network.valves.values(): 
            if 'Overflow' in valve.valve_id: 
                plt.plot(self.time, valve.h, label='Overflow')

        format_plot_time_axis(num_hours=24)

        plt.title("Overflow valve displacement")
        plt.xlabel(f'Time (h), dt = {self.dt}') 
        plt.ylabel('Valve displacement (-)') 
        plt.grid(True)
        plt.savefig(self.folder + '/overflow_h.png')


        if not plot: 
            plt.close(fig_overflow_T)
            plt.close(fig_overflow_mflow)
            plt.close(fig_overflow_h)


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

        tot_Q_d = np.zeros_like(self.time).astype(float)
        tot_Q_supply = np.zeros_like(self.time).astype(float)

        for hex_key in network.hexs.keys():
            hex = network.hexs[hex_key]
            tot_Q_d += hex.consumer.Q_d
            tot_Q_supply += hex.consumer.Q_supply

            plt.plot(self.time, hex.consumer.Q_d, label=f'Heat demand of C{hex.consumer.consumer_id.split(" ")[1]}')
            plt.plot(self.time, hex.consumer.Q_supply, label=f'Heat supplied to C{hex.consumer.consumer_id.split(" ")[1]}', linestyle='--')

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        plt.xlabel(f'Time (hours), dt = {self.dt}')
        plt.ylabel('Heat (W)')
        # plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/individual_heat.png')

        fig_tot = plt.figure()
        title = rf"Total Consumer Heat Demand vs Supply $\Delta Q$ = {np.sum(tot_Q_d - tot_Q_supply)/1e3:.3f} kJ"
        plt.title(title)
        plt.plot(self.time, tot_Q_d/1e3, label='Total heat demand')
        plt.plot(self.time, tot_Q_supply/1e3, label='Total heat supplied', linestyle='--')

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        plt.xlabel(f'Time (hours), dt = {self.dt}')
        plt.ylabel('Heat (kW)')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.folder + '/total_heat.png')

        fig_just_demand = plt.figure()
        plt.title("Total Consumer Heat Demand")
        plt.plot(self.time, tot_Q_d/1e3, label='Total heat demand')

        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        plt.xlabel(f'Time (hours), dt = {self.dt}')
        plt.ylabel('Heat (kW)')
        plt.grid(True)
        plt.savefig(self.folder + '/total_heat_demand.png')

        if not plot:
            plt.close(fig)
            plt.close(fig_tot)
            plt.close(fig_just_demand)

    def plot_h_valves(self, network: Network, plot = False):

        fig = plt.figure(figsize=(10, 6))
        plt.title("Valve displacement")
        for valve in network.valves.values():

            if 'Overflow' in valve.valve_id:
                label = 'Ov'
            else:
                label = valve.valve_id.split(" ")[1] 

            plt.plot(self.time, valve.h, label=label)      
    
        # Set x-axis to 0-24 hours (data stored in seconds). Show ticks every 4 hours.
        format_plot_time_axis(num_hours=24)

        plt.xlabel(f'Time (hours), dt = {self.dt}')
        plt.ylabel('Valve displacement (-)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.folder + '/valve_displacement.png')

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
        pipe_vflow_data = {}
        pipe_dp_data = {}
        pipe_dp_head_data = {}

        # For checking the temperature at supply and return risers
        pipe_supply_riser_temp = {}
        pipe_return_riser_temp = {}
                       
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
            pipe_vflow_data[f'{pipe_id}'] = np.round(pipe.mflow / (1000 * pipe.inner_cs),5)

            if 'Pump' in pipe_id:
                pipe_dp_data[f'{pipe_id} dp deliv'] = np.round(pipe.a * pipe.mflow**2 + pipe.b * pipe.mflow + pipe.c,5)
            else:
                pipe_dp_data[f'{pipe_id}'] = np.round(pipe.dp_friction_array,3)

            pipe_dp_head_data[f'{pipe_id}'] = np.round(pipe.pressure_elevation(),3)
            node_from = pipe_info['from']
            node_to = pipe_info['to']
            node_dT_data[f'dT {node_from.split()[1]}_{node_to.split()[1]}'] = np.round(network.nodes[node_from].T - network.nodes[node_to].T,3)

            if '.1' in pipe_id:  # Supply riser
                pipe_supply_riser_temp[pipe_id] = pipe.T
            if '.6' in pipe_id:  # Return riser
                pipe_return_riser_temp[pipe_id] = pipe.T

        # Save simulation data
        df_node = pd.DataFrame(node_data)
        df_node_dT = pd.DataFrame(node_dT_data)
        df_pipe_T = pd.DataFrame(pipe_T_data)
        df_pipe_mflow = pd.DataFrame(pipe_mflow_data)
        df_pipe_dp = pd.DataFrame(pipe_dp_data)
        df_pipe_dp_head = pd.DataFrame(pipe_dp_head_data, index = [0])
        df_pipe_vflow = pd.DataFrame(pipe_vflow_data)

        df_supply_riser_temp = pd.DataFrame(pipe_supply_riser_temp)
        df_return_riser_temp = pd.DataFrame(pipe_return_riser_temp) 

        df_node.to_csv(os.path.join(sim_data_folder, 'Node_temp.csv'), index=False)
        df_node_dT.to_csv(os.path.join(sim_data_folder,'Node_dT.csv'),index = False)
        df_pipe_T.to_csv(os.path.join(sim_data_folder,'Pipe_temp.csv'),index=False)
        df_pipe_mflow.to_csv(os.path.join(sim_data_folder,'Pipe_mflow.csv'), index =  False)
        df_pipe_dp.to_csv(os.path.join(sim_data_folder,'Pipe_dp_friction.csv'), index = False)
        df_pipe_dp_head.to_csv(os.path.join(sim_data_folder,'Pipe_dp_head.csv'), index = False)
        df_pipe_vflow.to_csv(os.path.join(sim_data_folder,'Pipe_vflow.csv'), index = False)
        
        df_supply_riser_temp.to_csv(os.path.join(sim_data_folder,'Supply_riser_temp.csv'), index = False)
        df_return_riser_temp.to_csv(os.path.join(sim_data_folder,'Return_riser_temp.csv'), index = False)
        
        # Pipe properties 
        filename = os.path.join(self.folder, 'simulation_data', "pipe_data.csv")

        rows = []   # temporary list of row dicts

        # Build row for each pipe
        for pipe_id, pipe_info in network.pipes.items():

            pipe = pipe_info['pipe_instance']

            row = {
                "pipe_id": pipe_id,
                "pipe_r_outer": pipe.r_outer,
                "pipe_r_inner": pipe.r_inner,
                "K": np.round(pipe.K,4),
                "rho_pipe": pipe.rho_pipe,
                "rho_insu": pipe.rho_insu,
                "cp_pipe": pipe.cp_pipe,
                "cp_insu": pipe.cp_insu,
                "insu_thickness": pipe.insu_thickness,
            }

            rows.append(row)

        # Convert to a small DataFrame
        df = pd.DataFrame(rows)

        # This overwrites the previous file completely
        df.to_csv(filename, index=False)

        # Saving the data corresponding to HEX and consumers
        HEX_data = {}
        hex_dp_data = {}
        hex_valve_data = {}
        overflow_data_normal = {}
        overflow_data_debug = {}

        total_heat = {}
        total_Q_demand = np.zeros_like(T_in)
        total_Q_supply = np.zeros_like(T_in)

        hex_folder = os.path.join(self.folder, 'hex_consumer_data')
        if not os.path.exists(hex_folder):
            os.makedirs(hex_folder)

        for valve_key in network.valves.keys():
            
            valve = network.valves[valve_key]
            hex_key = valve_key.replace("Valve", "Hex")

            if valve.hex is not None:

                hex = valve.hex
                
                HEX_data['Tc_in'] = hex.consumer.Tc_in
                HEX_data['Th_in'] = hex.pipes_in[f'Pipe {valve_key.split()[-1]}.3'].T
                HEX_data['Tc_out'] = hex.consumer.Tc_out
                HEX_data['Th_out'] = hex.T
                HEX_data['mflow_prim'] = hex.pipes_in[f'Pipe {valve_key.split()[-1]}.3'].mflow
                HEX_data['mflow_sec'] = hex.consumer.mflow           
                HEX_data['Q_d'] = hex.consumer.Q_d
                total_Q_demand += hex.consumer.Q_d
                HEX_data['Q_supply'] = hex.consumer.Q_supply
                total_Q_supply += hex.consumer.Q_supply
                HEX_data['h'] = valve.h
                HEX_data['Integral term'] = valve.I_array


                hex_mflow = hex.pipes_in[f'Pipe {valve_key.split()[-1]}.3'].mflow
                hex_dp_data[f'{valve_key}'] = (hex.Kp_rho_dp * hex_mflow**2).astype(int)

                hex_valve_data[f'Kv {valve_key}'] = valve.Kv
                hex_valve_data[f'h {valve_key}'] = valve.h

                dp = np.full_like(hex_mflow, np.nan, dtype=float)
                mask = valve.Kv != 0
                dp[mask] = (hex_mflow[mask] / valve.Kv[mask])**2
                hex_valve_data[f'dP {valve_key}'] = dp

                df_hex = pd.DataFrame(HEX_data)
                df_hex.to_csv(os.path.join(self.folder,'hex_consumer_data',f'{hex_key}.csv'), index = False)
                   
            else:
                overflow_pipe = network.pipes['Overflow 1']['pipe_instance']

                overflow_data_normal['Kv'] = valve.Kv
                overflow_data_normal['h'] = valve.h # actual h, with hstar taken into account
                overflow_data_normal['T_sensor'] = valve.T_sensor
                overflow_data_normal['T node'] = valve.node.T
                overflow_data_normal['mflow'] = overflow_pipe.mflow

                overflow_data_debug['Kv'] = valve.Kv
                overflow_data_debug['h'] = valve.h # actual h, with hstar taken into account
                overflow_data_debug['h_PI_lim'] = valve.h_PI_lim # change wanted by P-band
                overflow_data_debug['h_tau'] = valve.h_tau   # change slowed down by tau                
                overflow_data_debug['T_sensor'] = valve.T_sensor
                overflow_data_debug['T node'] = valve.node.T
                overflow_data_debug['T_pipe'] = overflow_pipe.T_pipe
                overflow_data_debug['T_pipe_lossless'] = overflow_pipe.T_lossless
                overflow_data_debug['T_pipe_in'] = overflow_pipe.T_in_extended[overflow_pipe.hist_len:]

                overflow_data_debug['mflow'] = overflow_pipe.mflow
                overflow_data_debug['t_stay'] = overflow_pipe.t_stay_array
                
                df_overflow = pd.DataFrame(overflow_data_normal)
                df_overflow.to_csv(os.path.join(self.folder,'hex_consumer_data','overflow.csv'), index = False)

                df_overflow_debug = pd.DataFrame(overflow_data_debug)
                df_overflow_debug.to_csv(os.path.join(self.folder,'hex_consumer_data','overflow_debug.csv'), index = False)
                
            df_hex_dp = pd.DataFrame(hex_dp_data)
            df_hex_dp.to_csv(os.path.join(self.folder,'hex_consumer_data','Hex_dp.csv'), index = False)

            df_hex_valve_data = pd.DataFrame(hex_valve_data)
            df_hex_valve_data.to_csv(os.path.join(self.folder,'hex_consumer_data','Hex_valve_data.csv'), index = False)

            total_heat["Demand"] = total_Q_demand
            total_heat["Supply"] = total_Q_supply

            df_total_heat = pd.DataFrame(total_heat)
            df_total_heat.to_csv(os.path.join(self.folder,'hex_consumer_data','total_heat.csv'), index = False)


        if network.theta is not None:
            theta_data = {
                'Min supply temperature [°C]': network.theta[0],
                'Max supply temperature [°C]': network.theta[1],
                'Heat demand threshold [W]': network.theta[2],
                'Heat demand P-band [W]': network.theta[3],
                'Overflow add T setpoint [°C]': network.theta[4],
                'Overflow P-band [°C]': network.theta[5]
            }
            df_theta = pd.DataFrame(theta_data, index = [0])
            df_theta.to_csv(os.path.join(self.folder,'hex_consumer_data','theta.csv'), index = False)



def format_plot_time_axis(num_hours=24):
    """
    Format plot time axis to show hours with ticks every 4 hours.
    
    Args:
        dt: time step in seconds
        num_hours: total time in hours (default 24)
    """
    ax = plt.gca()
    ax.set_xlim(0, num_hours * 3600)  # limits in seconds
    ticks_seconds = np.arange(0, num_hours + 1, 4) * 3600
    ax.set_xticks(ticks_seconds)
    ax.set_xticklabels([f'{int(h)}' for h in np.arange(0, num_hours + 1, 4)])
if __name__ == "__main__":
    pass 