from network import Network 
from simulation import Simulation
from scipy.signal import square
from sklearn.metrics import root_mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

class Test:

    def test_compare_bnode_method(number_of_nodes_array,
                                  temp_type,
                                  flow_type,
                                  dt,
                                  total_time,
                                  total_length,
                                  pipe_data_set):
        
        pipe_data, total_length, T_ambt = Test.read_pipe_data(pipe_data_set)
        plt.figure()
        plt.title("Comparision of #nodes in between for straight line")

        for number_of_nodes in number_of_nodes_array:

            print(f'number of nodes {number_of_nodes}')
            pipe_length = total_length / (number_of_nodes -1)

            net = Network('pipe split' + str(number_of_nodes))

            for node in range(1, number_of_nodes + 1):
                net.add_node('Node ' + str(node), 0, (node - 1)*pipe_length, 0)
            
            for node in range(1, number_of_nodes):
                net.add_pipe('Pipe ' + str(node), 'Node ' + str(node), 'Node ' + str(node + 1), pipe_data) 
                print("Pipe "+str(node))

            sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type, T_ambt)      

            T_in,v_flow = Test.generate_input(temp_type, flow_type, total_time, dt)

            sim.simulate_network(net, T_in, v_flow, T_ambt)


            end_Node_T = net.nodes['Node ' + str(number_of_nodes)].T

            plt.plot(sim.time, end_Node_T, label = '#'+str(number_of_nodes - 2))        
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [°C]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def simulate_network_compare(network,
                        T_ambt,
                        dt,
                        total_time = None,
                        file = None, 
                        temp_type = None,
                        flow_type = None,
                        plot_nodes_T = False, 
                        plot_pipes_T = False, 
                        plot_pipes_m_flow = False, 
                        plot_network = False,
                        plot_nodes_dT = False,
                        no_cap = False):
        

        # In case of real data
        if file != None:
            
            # Load data
            basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            data_csv = pd.read_csv(os.path.join(basedir, 'data', 'pipe_validation', 'experiment', file + '_interpolated.csv'))
            T_in = data_csv['InletWaterTemp'].values
            m_flow = data_csv['MassFlowRate'].values
            T_init_water = data_csv['OutletWaterTemp'].values[0] # initial water temperature in the network
            # T_init_pipe = data_csv['OutletPipeTemp'].values[0] # initial pipe temperature in the network
            T_init_pipe = T_init_water
           
            # Step through data to create smaller vectors at dt intervals
            total_time = len(T_in)
            T_in = T_in[::dt]
            m_flow = m_flow[::dt]
            

            pipe1 = network.pipes['Pipe 1']['pipe_instance']
            v_flow = np.round(m_flow / pipe1.rho_water / pipe1.inner_cs, 3)       #TODO Temporary solution for now the mass flow data. Maybe later I should reconstruct the code 
            sim = Simulation(dt, total_time, network.net_id, T_ambt, file = file, no_cap = no_cap)      

        else:
            # In case of synthetic data
            T_in, v_flow = Test.generate_input(temp_type, flow_type, total_time, dt)
            sim = Simulation(dt, total_time, network.net_id, T_ambt, temp_type = temp_type, flow_type = flow_type, no_cap = no_cap)
            T_init_water = T_init_pipe = T_ambt      

        sim.simulate_network(network, T_in, v_flow, T_init_water, T_init_pipe, T_ambt, 
                                plot_network = plot_network, 
                                plot_nodes_T = plot_nodes_T, 
                                plot_pipes_T = plot_pipes_T, 
                                plot_pipes_m_flow = plot_pipes_m_flow, 
                                plot_nodes_dT = plot_nodes_dT,
                                no_cap = no_cap)

    def compare_simulations(network,
                            T_ambt,
                            dt,
                            total_time = None,
                            file = None,
                            temp_type = None,
                            flow_type = None,
                            number_of_nodes = None,
                            no_cap = False):
     
        """
        Function for comparing the thermodynamic pipeline models from Modelica and the Node Method, with each other and experimental data.
        Args: 
            - network : compared network
            - T_ambt : ambient temperature
            - total_time: only necessary when non experimental values.

        Number of nodes only important for Node Method vs Modelica. As it concerns the number of nodes in the finite volume method.
        Therefore the experiment doesn't require it. 
        """

        base_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dict = {}
        
        # Perform the simulation
        Test.simulate_network_compare(network,T_ambt, dt, total_time, file, temp_type, flow_type, no_cap = no_cap)

        if file:

            # To set correct folder
            exp_csv = pd.read_csv(os.path.join(base_dir, 'data', 'pipe_validation', 'experiment', file + '_interpolated.csv'))
            total_time = len(exp_csv['MassFlowRate']) - 1 #because file starts at 0
            temp_type = flow_type = file

            T_out_exp = exp_csv['OutletWaterTemp'].values
            m_flow_exp = exp_csv['MassFlowRate'].values
            
            data_dict['Exp temp'] = T_out_exp[::dt]

            sim_name = f"{file}_dt={dt}_Tambt={T_ambt}"
            mo_name = f"{file}_dt={dt}_Tambt={T_ambt}_mo_clean.csv"
            
        
        else:
            sim_name = (f"network={network.net_id}_dt={dt}_total_time={total_time}_"
                f"Tin={temp_type}_mflow={flow_type}_Tambt={T_ambt}")
            
            total_length = int(network.get_total_network_length())
            mo_name = (f"{total_length}m_dt={dt}_Tin={temp_type}_mflow={flow_type}_"
                    f"Tambt={T_ambt}_nodes={number_of_nodes}_mo")

            if no_cap:
                sim_name += "_no_cap" 
                mo_name += "_no_cap"
                print("Rembemer: heat capacity in modelica still activated!")
            
            mo_name += "_clean.csv"
            
        # Path names
        sim_folder = os.path.join(base_dir,'figures','simulation',sim_name)                               
        sim_file = os.path.join(sim_folder, 'simulation_data.csv')
        mo_file = os.path.join(base_dir, "data", "pipe_validation", "modelica", mo_name)

              
        # Simulation and modelica data
        sim_data = pd.read_csv(sim_file)
        sim_temp = sim_data[f"T_Node {len(network.nodes)}"]

        mo_data = pd.read_csv(mo_file, delimiter=",")

        # Experimental data is all saved at 1 sec, when synthetic data is used, it is saved at dt sec
        if file:
            mo_temp = mo_data['T_sensor2.T'][::dt]
            mo_Tin = mo_data['T_sensor1.T'][::dt]
            mo_time = mo_data['time'][::dt]          

        else:            
        # The original modelica data is saved in a fishy manner, where you have a lot of duplicates and not always at the correct dt.
        # Therefore to still make the modelica values appear at correct time, I use the modelica time and subtract the last value.
            mo_temp = mo_data['T_sensor2.T'][:-1]
            mo_Tin = mo_data['T_sensor1.T'][:-1]
            mo_time = mo_data['time'][:-1]
        
        plots_folder = os.path.join(
                            base_dir,
                            "figures",
                            "validation",
                            "exp_simulation_modelica" if file else "simulation_modelica",
                            sim_name if file else f'{sim_name}_nodes={number_of_nodes}',
                        )
        
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        plt.figure()
        if file:
            plt.plot(T_out_exp, label = "Experimental data")
            plt.plot(mo_time, mo_temp, label = f'Modelica RMSE {round(root_mean_squared_error(T_out_exp[::dt],mo_temp),2)}')
            plt.plot(sim_data['time'], sim_temp, label = f'Simulation RMSE {round(root_mean_squared_error(T_out_exp[::dt], sim_temp),2)}')
            plt.title(f"Temperature comparison: Experiment {file[-1]} ")
        else:
            plt.title(f"Temperature comparison Tin: {temp_type}, mass flow: {flow_type} ")
            plt.plot(mo_time, mo_temp, label = f'Modelica, #nodes = {number_of_nodes}')
            plt.plot(sim_data['time'], sim_temp, label = f'Simulation RMSE {round(root_mean_squared_error(mo_temp, sim_temp),2)}')
        plt.plot(mo_time,mo_Tin, label = 'Inlet temperature',linestyle = '--')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_folder, 'temperature_comparison.png'))
        plt.close()
        
        plt.figure()
        plt.title("Input mass flow rate")
        plt.plot(sim_data['time'], sim_data['m_flow Pipe 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Mass Flow (kg/s)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_folder, 'inlet_mass_flow.png'))
        plt.close()

        # Save data
        data_dict.update({
            "Time": sim_data['time'],
            "Simulation temp": sim_temp.values,
            "Modelica temp": mo_temp.values,
            "Mass flow": sim_data["m_flow Pipe 1"].values,
            "Input temp simulation": sim_data["T_Node 1"].values,
            "Input temp modelica": mo_Tin.values,
        })

        df = pd.DataFrame(data_dict)
        cols = ["Time"] + [col for col in df.columns if col != "Time"]
        df = df[cols]
        df.to_csv(os.path.join(plots_folder, "comparison_data.csv"), index=False)

###########################################################
# Help functions for the tests
###########################################################

    def network_builder_one_pipe(pipe_data_set,
                        number_of_nodes, total_length):
        
        pipe_data = Test.read_pipe_data(pipe_data_set)
        number_of_nodes = 2
        pipe_length = total_length / (number_of_nodes -1)

        net = Network("One_pipe_#nodes_" + str(number_of_nodes) + "_length=" + str(total_length))
        for node in range(1, number_of_nodes + 1):
            net.add_node('Node ' + str(node), 0, (node - 1)*pipe_length, 0)
        
        for node in range(1, number_of_nodes):
            net.add_pipe('Pipe ' + str(node), 'Node ' + str(node), 'Node ' + str(node + 1), pipe_data) 
        return net

    def read_pipe_data(pipe_data_set):

        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        r_inner = constants[pipe_data_set]['r_inner'] # [m]
        r_outer = constants[pipe_data_set]['r_outer'] # [m]
        # K = constants[pipe_data_set]['K']
        rho_pipe = constants[pipe_data_set]['rho_pipe'] # [kg / m3]
        cp_pipe = constants[pipe_data_set]['cp_pipe']  # [J / kg K]
        rho_insu = constants[pipe_data_set]['rho_insu'] # [kg / m3]
        cp_insu = constants[pipe_data_set]['cp_insu']  # [J/ kg K]
        insu_thickness = constants[pipe_data_set]['insu_thickness'] # [m]

        k_pipe = constants[pipe_data_set]['k_pipe'] #thermal conductivity pipe [W / m K]
        k_insu = constants[pipe_data_set]['k_insu'] #thermal conductivity insulation [W / m K]
        h_pipe_air = constants[pipe_data_set]['h_pipe_air'] # natural convection from pipe to surrounding air [W / m2 K]

        R = (
                np.log(r_outer/r_inner)/(2*np.pi*k_pipe) # conduction pipe
                + np.log((r_outer + insu_thickness)/r_outer)/(2*np.pi*k_insu) # conduction insulation
                + 1/(2*np.pi*(r_outer+insu_thickness)*h_pipe_air) # convection pipe to air
            )

        K = 1/R # total thermal conductivity [W / m K]

        pipe_data = [r_inner, r_outer, cp_pipe, rho_pipe, cp_insu, rho_insu, insu_thickness, K]

        return pipe_data
    
    def generate_input(temp_type, flow_type, total_time, dt):

        """
        Generate time series for inlet temperature and flow velocity used in simulations.

        Args:
            temp_type (str): Type of temperature input. One of:
                - "constant": constant inlet temperature
                - "oscillation": smooth sinusoidal variation
                - "square": square-wave variation
            flow_type (str): Type of flow velocity input. One of:
                - "constant": constant flow velocity
                - "oscillation": smooth oscillating flow velocity
                - "square": square-wave flow velocity
            total_time (float|int): Total simulated time in seconds.
            dt (float|int): Time step in seconds.

        Returns:
            tuple of np.ndarray:
                - T_in: inlet temperature time series, length = num_steps
                - v_flow: flow velocity time series, length = num_steps

        """

        # Check if total_time / dt is a whole number
        if (total_time / dt) % 1 != 0:
                    num_steps = int(total_time / dt) + 1
        else: 
            num_steps = int(total_time/dt)


        if temp_type == "constant":
            T_in = np.ones(num_steps) * 65                                 # Constant
        elif temp_type == "oscillation":
            T_in = 65 + 5 * np.sin(np.linspace(0, 8*np.pi, num_steps))   # Oscillating inlet temperature
        elif temp_type == "square":
            T_in = 80 + 1* square(2 * np.pi * total_time / 20)       
        else:
            raise ValueError("This temperature type doesn't exist!")
    

        if flow_type == "constant":
            v_flow = np.ones(num_steps) * 2                           # Constant
        elif flow_type == "oscillation":
            v_flow = 1.5 + 0.8*np.cos(np.linspace(0, 2*np.pi, num_steps)) # Oscillating flow velocity
        elif flow_type == "square":
            v_flow = 1.5 + 0.5 * square(2 * np.pi * total_time / 50)        # Square wave flow velocity, 50 is the period
        else:
            raise ValueError("This flow type doesn't exist!")
        return T_in, v_flow
    
if __name__ == "__main__":

    # files = ['ExperimentA', 'ExperimentB', 'ExperimentC', 'ExperimentD']
    # dt_array = [1,1,1,30] # [s], delta time for every file

    # number_of_nodes = 2
    # T_ambt = 18 # [°C] Staat nu nog ook in de file van van der Heijden! MOET NAAR 18, MAAR EERST DAARVOOR MODELICA RUNNEN
    # total_length = 39 # [m]

    # network_exp = Test.network_builder_one_pipe('Pipe of experiment van der Heijden', number_of_nodes, total_length)
    # # # Test.compare_simulations(network_exp, T_ambt, dt_array[0], file = files[0])
    # for k in range(len(files)):
    #     Test.compare_simulations(network_exp, T_ambt, dt_array[k], file = files[k], no_cap = False)

    # dt = 1 # [s]
    # total_L = 40
    # total_time = 160
    # T_ambt = 20
    # nodes = [25,50,100,200]

    network_synt = Test.network_builder_one_pipe('Pipe of experiment van der Heijden', 2, total_L)

    # # Test.simulate_network(network_synt,5, dt, total_time = 8000, temp_type = 'oscillation', flow_type = 'constant')
    # # With oscillation T, constant Flow
    # for node in nodes:
    #     Test.compare_simulations(network_synt, T_ambt, dt, total_time = 8000, temp_type = 'oscillation', flow_type = 'constant',number_of_nodes=node)

    # Test.compare_simulations(network_synt,T_ambt,dt,total_time,temp_type = 'oscillation',flow_type = 'constant', number_of_nodes=80)

    
    

    