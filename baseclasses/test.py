from network import Network 
from simulation import Simulation
from consumer import Consumer

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
                        plot_pipes_mflow = False, 
                        plot_network = False,
                        plot_nodes_dT = False,
                        no_cap = False):
        

        # In case of real data
        if file != None:
            
            # Load data
            basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            data_csv = pd.read_csv(os.path.join(basedir, 'data', 'pipe_validation', 'experiment', file + '_interpolated.csv'))
            T_in = data_csv['InletWaterTemp'].values
            mflow = data_csv['MassFlowRate'].values
            T_init_water = data_csv['OutletWaterTemp'].values[0] # initial water temperature in the network
            # T_init_pipe = data_csv['OutletPipeTemp'].values[0] # initial pipe temperature in the network
            T_init_pipe = T_init_water
           
            # Step through data to create smaller vectors at dt intervals
            total_time = len(T_in)
            T_in = T_in[::dt]
            mflow = mflow[::dt]
            

            pipe1 = network.pipes['Pipe 1']['pipe_instance']
            v_flow = np.round(mflow / pipe1.rho_water / pipe1.inner_cs, 3)       #TODO Temporary solution for now the mass flow data. Maybe later I should reconstruct the code 
            sim = Simulation(dt, total_time, network.net_id, T_ambt, file = file, no_cap = no_cap)      

        else:
            # In case of synthetic data
            T_in, v_flow = Test.generate_input(temp_type, flow_type, total_time, dt)
            sim = Simulation(dt, total_time, network.net_id, T_ambt, temp_type = temp_type, flow_type = flow_type, no_cap = no_cap)
            T_init_water = T_init_pipe = T_ambt      

        sim.simulate_network(network, T_in, v_flow, T_init_water, T_init_pipe, 
                                plot_network = plot_network, 
                                plot_nodes_T = plot_nodes_T, 
                                plot_pipes_T = plot_pipes_T, 
                                plot_pipes_mflow = plot_pipes_mflow, 
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
            - file : name of the experimental file. If None, synthetic data is used. And the name of the simulation folder will be based on that. 

        Number of nodes only important for Node Method vs Modelica. As it concerns the number of nodes in the finite volume method.
        Therefore the experiment doesn't require it. 
        """

        base_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dict = {}
        
        # Perform the simulation
        Test.simulate_network_compare(network,T_ambt, dt, 
                                      total_time = total_time,
                                      file = file,
                                      temp_type = temp_type,
                                      flow_type = flow_type,
                                      no_cap = no_cap)

        if file:

            # To set correct folder
            exp_csv = pd.read_csv(os.path.join(base_dir, 'data', 'pipe_validation', 'experiment', file + '_interpolated.csv'))
            total_time = len(exp_csv['MassFlowRate']) - 1 #because file starts at 0
            temp_type = flow_type = file

            T_out_exp = exp_csv['OutletWaterTemp'].values
            mflow_exp = exp_csv['MassFlowRate'].values
            
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
        sim_folder = os.path.join(sim_folder,'simulation_data')
        mo_file = os.path.join(base_dir, "data", "pipe_validation", "modelica", mo_name)

              
        # Simulation and modelica data
        sim_data_mflow = pd.read_csv(os.path.join(sim_folder, 'Pipe_mflow.csv'))
        sim_data_temp = pd.read_csv(os.path.join(sim_folder, 'Node_temp.csv'))

        sim_temp = sim_data_temp[f"Node {len(network.nodes)}"]

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
                            "pipe_validation",
                            "exp_simulation_modelica" if file else "simulation_modelica",
                            sim_name if file else f'{sim_name}_nodes={number_of_nodes}',
                        )
        
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        plt.figure()
        if file:
            plt.plot(T_out_exp, label = "Experimental data")
            plt.plot(mo_time, mo_temp, label = f'Modelica RMSE {round(root_mean_squared_error(T_out_exp[::dt],mo_temp),2)}')
            plt.plot(sim_data_temp['time'], sim_temp, label = f'Simulation RMSE {round(root_mean_squared_error(T_out_exp[::dt], sim_temp),2)}')
            plt.title(f"Temperature comparison: Experiment {file[-1]} ")
        else:
            plt.title(f"Temperature comparison Tin: {temp_type}, mass flow: {flow_type} ")
            plt.plot(mo_time, mo_temp, label = f'Modelica, #nodes = {number_of_nodes}')
            plt.plot(sim_data_temp['time'], sim_temp, label = f'Simulation RMSE {round(root_mean_squared_error(mo_temp, sim_temp),2)}')
        plt.plot(mo_time,mo_Tin, label = 'Inlet temperature',linestyle = '--')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_folder, 'temperature_comparison.png'))
        plt.close()
        
        plt.figure()
        plt.title("Input mass flow rate")
        plt.plot(sim_data_temp['time'], sim_data_mflow['Pipe 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Mass Flow (kg/s)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_folder, 'inlet_mass_flow.png'))
        plt.close()

        # Save data
        data_dict.update({
            "Time": sim_data_temp['time'],
            "Simulation temp": sim_temp.values,
            "Modelica temp": mo_temp.values,
            "Mass flow": sim_data_mflow["Pipe 1"].values,
            "Input temp simulation": sim_data_temp["Node 1"].values,
            "Input temp modelica": mo_Tin.values,
        })

        df = pd.DataFrame(data_dict)
        cols = ["Time"] + [col for col in df.columns if col != "Time"]
        df = df[cols]
        df.to_csv(os.path.join(plots_folder, "comparison_data.csv"), index=False)

    def initial_test_HEX():
        # Create consumer
        # Values chosen to mimic realistic heat demand profile based on literature. 
        # And the integral of the heat demand is scaled to 65 MJ/day. 
         
        A1 = 0.109*1.524
        A2 = 0.113*1.524
        Period1 = 2*np.pi / 0.298
        Period2 = 2*np.pi / 0.529
        Phi1 = -1.949
        Phi2 = -2.154
        offset = 0.509*1.524
        tau = 0 

        consumer = Consumer('Consumer 1',A1,A2,Period1,Period2,Phi1,Phi2,offset,tau)

        pipe_data = Test.read_pipe_data('Pipe of experiment van der Heijden')
        hex_data = Test.read_hex_data('Standard hex constants dummy pressure')

        # Create network
        net = Network("initial test HEX")

        net.add_node('Node 1', 0, 0, 0)
        net.add_node('Node 2', 0, 0, 6)
                
        net.add_hex('Hex 1', 'Node 1', 'Node 2', hex_data, pipe_data, consumer)

        # Simulation parameters
        dt = 60 # s
        total_time = 24 * 3600 # h
        T_ambt = 20

        # Input profiles
        temp_type = 'constant'
        flow_type = 'constant'
        
        T_in, v_flow = Test.generate_input_one_pipe(temp_type,flow_type, total_time, dt)

        # Run simulation
        sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type, flow_type = flow_type)      
        sim.simulate_network(net, T_in, v_flow, T_ambt, T_ambt, T_ambt)

    def model_step_5():
        """
        Initial test to see whether the heat exchanger class is working properly.
        The consumer demand profile is set for 1 day. 
        """
        # Create consumer
        # Values chosen to mimic realistic heat demand profile based on literature. 
        # And the integral of the heat demand is scaled to 65 MJ/day. 
         
        A1 = 0.109*1.524
        A2 = 0.113*1.524
        Period1 = 2*np.pi / 0.298
        Period2 = 2*np.pi / 0.529
        Phi1 = -1.949
        Phi2 = -2.154
        offset = 0.509*1.524
        tau = 3600 

        consumer1 = Consumer('Consumer 1',A1,A2,Period1,Period2,Phi1,Phi2,offset,0)
        consumer2 = Consumer('Consumer 2',A1,A2,Period1,Period2,Phi1,Phi2,offset,tau)

        pipe_data = Test.read_pipe_data('DN40')
        hex_data = Test.read_hex_data('Standard hex constants dummy pressure')

        # Create network
        net = Network("Model step 5")

        net.add_node('Node 1', 0, 0, 0)
        net.add_node('Node 2', 0, 0, 6)
        net.add_node('Node 3', 0, 0, 12)
        net.add_node('Node 4', 0, 0, 13)
        net.add_node('Node 5', 2, 0, 13)
        net.add_node('Node 6', 5, 0, 12)
        net.add_node('Node 7', 5, 0, 11)
        net.add_node('Node 8', 2, 0, 11)
        net.add_node('Node 9', 5, 0, 6)
        net.add_node('Node 10', 5, 0, 5)
        net.add_node('Node 11', 2, 0, 5)
        net.add_node('Node 12',2, 0, 0)

        net.add_pipe('Pipe 1', 'Node 1', 'Node 2', pipe_data)
        net.add_pipe('Pipe 2', 'Node 2', 'Node 3', pipe_data)
        net.add_pipe('Pipe 3', 'Node 3', 'Node 4', pipe_data)
        net.add_pipe('Pipe 4', 'Node 4', 'Node 5', pipe_data)
        net.add_pipe('Pipe 5', 'Node 5', 'Node 8', pipe_data)
        net.add_pipe('Pipe 6', 'Node 3', 'Node 6', pipe_data)
        net.add_pipe('Pipe 7', 'Node 7', 'Node 8', pipe_data)
        net.add_pipe('Pipe 8', 'Node 2', 'Node 9', pipe_data)
        net.add_pipe('Pipe 9', 'Node 8', 'Node 11', pipe_data)
        net.add_pipe('Pipe 10', 'Node 10', 'Node 11', pipe_data)
        net.add_pipe('Pipe 11', 'Node 11', 'Node 12', pipe_data)
        
        net.add_hex('Hex 1', 'Node 6', 'Node 7', hex_data, pipe_data, consumer1)
        net.add_hex('Hex 2', 'Node 9', 'Node 10', hex_data, pipe_data, consumer2)

        # Simulation parameters
        dt = 60 # s
        total_time = 24 * 3600 # h
        T_ambt = 20

        # Input profiles
        temp_type = 'constant'
        flow_type = 'constant'
        
        T_in, v_flow = Test.generate_input_network(temp_type,flow_type, total_time, dt)

        # Run simulation
        sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type, flow_type = flow_type)      
        sim.simulate_network(net, T_in, v_flow, T_ambt, T_ambt, T_ambt)

    def model_step_7():
        """
        Initial test to see whether the incidence and loop matrix construction is working properly.
        """
        pipe_data_DN40 = Test.read_pipe_data('DN40')
        pipe_data_DN20 = Test.read_pipe_data('DN20')
        hex_data = Test.read_hex_data('Standard hex constants')
        pump_data = Test.read_pump_data('50kPa Pump constant')
     
        # Create network
        net = Network("Model step 7")
        number_consumers = 5
        pipe_data_list = [pipe_data_DN40] * number_consumers
        Test.network_builder(net, pipe_data_list, pipe_data_DN20, hex_data, pump_data, number_consumers, use_overflow = False)

        # Simulation parameters
        dt = 60 # s
        total_time = 24 * 3600 # h
        T_ambt = 20

        # Input profiles
        temp_type = 'constant'
        
        T_in = Test.generate_input_network(temp_type, total_time, dt)

        # Run simulation
        sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)
        # sim.plot_network(net)      
        sim.simulate_network(net, T_in, T_ambt, T_ambt)
    
    def model_network_Rutger():
        
        """
        Replicate network of whcih Rutger send data from
        """

        pipe_data_DN20 = Test.read_pipe_data('DN20')
        pipe_data_DN25 = Test.read_pipe_data('DN25')
        pipe_data_DN32 = Test.read_pipe_data('DN32')
        pipe_data_DN40 = Test.read_pipe_data('DN40')

        hex_data = Test.read_hex_data('Standard hex constants')
        pump_data = Test.read_pump_data('50kPa Pump constant')
     
        # Create network
        net = Network("Network Rutger")
        
        number_consumers = 23
        pipe_data_list = [pipe_data_DN40] * 6 +[pipe_data_DN32] * 14 + [pipe_data_DN25] * 3
        
        Test.network_builder(net, pipe_data_list, pipe_data_DN20, hex_data, pump_data, number_consumers, use_overflow = False)

        # Simulation parameters
        dt = 60 # s
        total_time = 24 * 3600 # h
        T_ambt = 20

        # Input profiles
        temp_type = 'constant'
        
        T_in = Test.generate_input_network(temp_type, total_time, dt)

        # Run simulation
        sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)
        sim.simulate_network(net, T_in, T_ambt, T_ambt)

    def test_incidence_and_loop_matrices():
        
        pipe_data = Test.read_pipe_data('Pipe of DN40')
        hex_data = Test.read_hex_data('Standard hex constants dummy pressure')

        A1 = 0.109*1.524
        A2 = 0.113*1.524
        Period1 = 2*np.pi / 0.298
        Period2 = 2*np.pi / 0.529
        Phi1 = -1.949
        Phi2 = -2.154
        offset = 0.509*1.524
        tau = 3600 

        consumer1 = Consumer('Consumer 1',A1,A2,Period1,Period2,Phi1,Phi2,offset,0)
        consumer2 = Consumer('Consumer 2',A1,A2,Period1,Period2,Phi1,Phi2,offset,tau)

        # Create network
        net = Network("Model step 5")
     
        net.add_node('Node 1', 0, 0, 0)
        net.add_node('Node 2', 0, 0, 6)
        net.add_node('Node 3', 0, 0, 12)
        net.add_node('Node 4', 0, 0, 13)
        net.add_node('Node 5', 2, 0, 13)
        net.add_node('Node 6', 5, 0, 12)
        net.add_node('Node 7', 5, 0, 11)
        net.add_node('Node 8', 2, 0, 11)
        net.add_node('Node 9', 5, 0, 6)
        net.add_node('Node 10', 5, 0, 5)
        net.add_node('Node 11', 2, 0, 5)
        net.add_node('Node 12',2, 0, 0)

        net.add_pump('Pump 1', 'Node 12', 'Node 1', pipe_data, 2e5)

        net.add_pipe('Pipe 1', 'Node 1', 'Node 2', pipe_data)
        net.add_pipe('Pipe 2', 'Node 2', 'Node 3', pipe_data)
        net.add_pipe('Pipe 3', 'Node 3', 'Node 4', pipe_data)
        net.add_pipe('Pipe 4', 'Node 4', 'Node 5', pipe_data)
        net.add_pipe('Pipe 5', 'Node 5', 'Node 8', pipe_data)
        net.add_pipe('Pipe 6', 'Node 3', 'Node 6', pipe_data)
        net.add_pipe('Pipe 7', 'Node 7', 'Node 8', pipe_data)
        net.add_pipe('Pipe 8', 'Node 2', 'Node 9', pipe_data)
        net.add_pipe('Pipe 9', 'Node 8', 'Node 11', pipe_data)
        net.add_pipe('Pipe 10', 'Node 10', 'Node 11', pipe_data)
        net.add_pipe('Pipe 11', 'Node 11', 'Node 12', pipe_data)
        
        net.add_hex('Hex 1', 'Node 6', 'Node 7', hex_data, pipe_data, consumer1)
        net.add_hex('Hex 2', 'Node 9', 'Node 10', hex_data, pipe_data, consumer2)

        net.build_incidence_matrix()
        # print(net.incidence_matrix)
        net.build_loop_matrix_from_incidence()
        print(net.loop_matrix)

        pipe_ids = list(net.pipes.keys())

        for i in range(len(net.loop_matrix)):
            loop = []
            for j in range(len(net.loop_matrix[i])):
                if net.loop_matrix[i][j] == 1:
                    loop.append(pipe_ids[j] + " (+)")
                elif net.loop_matrix[i][j] == -1:
                    loop.append(pipe_ids[j] + " (-)")
            print(f"Loop {i+1}: " + ", ".join(loop))

    def test_NR():

        pipe_data = Test.read_pipe_data('DN40')
        pipe_data_hex = Test.read_pipe_data('DN20')
        hex_data = Test.read_hex_data('Standard hex constants')

        # Create network
        net = Network("NR test")

        A1 = 0.109*1.524
        A2 = 0.113*1.524
        Period1 = 2*np.pi / 0.298
        Period2 = 2*np.pi / 0.529
        Phi1 = -1.949
        Phi2 = -2.154
        offset = 0.509*1.524
        tau = 3600 

        consumer1 = Consumer('Consumer 1',A1,A2,Period1,Period2,Phi1,Phi2,offset,0)
     
        net.add_node('Node 1', 0, 0, 0)
        net.add_node('Node 2', 0, 3, 0)
        net.add_node('Node 3', 3, 3, 0)
        net.add_node('Node 4', 3, 0, 0)


        net.add_pipe('Pipe 1', 'Node 1', 'Node 2', pipe_data)
        net.add_pipe('Pipe 2', 'Node 2', 'Node 3', pipe_data)
        # net.add_pipe('Pipe 3', 'Node 3', 'Node 4', pipe_data)
        net.add_hex('Hex 1', 'Node 3', 'Node 4', hex_data, pipe_data_hex, consumer1)
        net.add_pump('Pump 1', 'Node 4', 'Node 1', pipe_data, 4e4)

        # Simulation parameters
        dt = 60 # s
        total_time = 24 * 3600 # h
        T_ambt = 20

        # Input profiles
        temp_type = 'constant'
        
        T_in = Test.generate_input_network(temp_type, total_time, dt)

        # Run simulation
        sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)      
        sim.simulate_network(net, T_in, T_ambt, T_ambt)

    def test_network_builder():

        pipe_data_DN40 = Test.read_pipe_data('DN40')
        pipe_data_DN20 = Test.read_pipe_data('DN20')
        hex_data = Test.read_hex_data('Standard hex constants')
        pump_data = Test.read_pump_data('50kPa Pump constant')
     
        # Create network
        net = Network("Network builder")
        number_consumers = 5
        pipe_data_list = [pipe_data_DN40] * number_consumers

        Test.network_builder(net, pipe_data_list,pipe_data_DN20, hex_data,pump_data,number_consumers)

        # Simulation parameters
        dt = 60 # s
        total_time = 24 * 3600 # h
        T_ambt = 20

        # Input profiles
        temp_type = 'constant'
        
        # Run simulation
        sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)
        sim.plot_network(net) 

###########################################################
# Help functions for the tests
###########################################################

    def network_builder_one_pipe(pipe_data_set,
                        number_of_nodes, total_length):
        
        pipe_data = Test.read_pipe_data(pipe_data_set)
        pipe_length = total_length / (number_of_nodes -1)

        net = Network("One_pipe_#nodes_" + str(number_of_nodes) + "_length=" + str(total_length))
        for node in range(1, number_of_nodes + 1):
            net.add_node('Node ' + str(node), 0, (node - 1)*pipe_length, 0)
        
        for node in range(1, number_of_nodes):
            net.add_pipe('Pipe ' + str(node), 'Node ' + str(node), 'Node ' + str(node + 1), pipe_data) 
        return net

    def read_pipe_data(pipe_data_set):

        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(os.path.dirname(thesis_dir),'constants', 'constants_pipe.json')

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

        k_pipe = constants[pipe_data_set]['k_pipe'] # thermal conductivity pipe [W / m K]
        k_insu = constants[pipe_data_set]['k_insu'] # thermal conductivity insulation [W / m K]
        h_pipe_air = constants[pipe_data_set]['h_pipe_air'] # natural convection from pipe to surrounding air [W / m2 K]

        epsilon = constants[pipe_data_set]['epsilon'] # rougness of inner pipe [m]

        R = (
                np.log(r_outer/r_inner)/(2*np.pi*k_pipe) # conduction pipe
                + np.log((r_outer + insu_thickness)/r_outer)/(2*np.pi*k_insu) # conduction insulation
                + 1/(2*np.pi*(r_outer+insu_thickness)*h_pipe_air) # convection pipe to air
            )

        K = 1/R # total thermal conductivity [W / m K]

        pipe_data = [r_inner, r_outer, cp_pipe, rho_pipe, cp_insu, rho_insu, insu_thickness, K, epsilon]

        return pipe_data
    
    def read_hex_data(hex_data_set):

        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(os.path.dirname(thesis_dir),'constants', 'constants_hex.json')

        with open(constants_file) as f:
            constants = json.load(f)

        U = constants[hex_data_set]['U'] # Overall heat transfer coefficient [W/m2K]
        As = constants[hex_data_set]['As'] # Heat transfer area [m2]
        F = constants[hex_data_set]['F'] # Correction factor [-]
        Kp = constants[hex_data_set]['Kp'] # Pressure loss coefficient [-]
        K_vs = constants[hex_data_set]['K_vs'] # Hydrualic conductivity for valve [m3/s Pa^0.5]
        K_v0 = constants[hex_data_set]['K_v0'] # Hydraulic conductivity for valve when closed[m3/s Pa^0.5]

        hex_data = [U, As, F, Kp, K_vs, K_v0]

        return hex_data
    
    def read_pump_data(pump_data_set):

        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(os.path.dirname(thesis_dir),'constants', 'constants_pump.json')

        with open(constants_file) as f:
            constants = json.load(f)

        a = constants[pump_data_set]['a'] # Pump curve coefficients [a,b,c] for a quadratic curve
        b = constants[pump_data_set]['b'] # 
        c = constants[pump_data_set]['c'] #

        pump_data = [a,b,c]

        return pump_data
    
    def generate_input_one_pipe(temp_type, flow_type, total_time, dt):

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
    
    def generate_input_network(temp_type, total_time, dt):

        """
        Generate time series for inlet temperature used in simulations.
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

        return T_in
    
    def network_builder(net : Network, 
                        pipe_data_list : dict, 
                        pipe_hex_data, 
                        hex_data, 
                        pump_data,
                        number_of_consumers,
                        use_overflow = True):
        
        # Add heat exchangers and connecting pipes based on number of consumers
        A1 = 0.109*1.524
        A2 = 0.113*1.524
        Period1 = 2*np.pi / 0.298
        Period2 = 2*np.pi / 0.529
        Phi1 = -1.949
        Phi2 = -2.154
        offset = 0.509*1.524
        tau = 0 

        consumer = Consumer('Consumer 1', A1, A2, Period1, Period2, Phi1, Phi2, offset, tau) 

        net.add_node('Node 1', 0, 0, 0)
        net.add_node('Node 2', 0, 0, 3)
        net.add_node('Node 3', 2, 0, 3)
        net.add_node('Node 4', 2, 0, 2)
        net.add_node('Node 5', 1, 0, 2)
        net.add_node('Node 6', 1, 0, 0)

        net.add_pump('Pump 1', 'Node 6', 'Node 1', pipe_data_list[0], pump_data)
        net.add_pipe('Pipe 1','Node 1', 'Node 2', pipe_data_list[0])
        net.add_pipe('Pipe 2', 'Node 2', 'Node 3', pipe_data_list[0])
        net.add_pipe('Pipe 3', 'Node 4', 'Node 5', pipe_data_list[0])
        net.add_pipe('Pipe 4', 'Node 5', 'Node 6', pipe_data_list[0])
                
        net.add_hex('Hex 1', 'Node 3', 'Node 4', hex_data, pipe_hex_data, consumer)
        
        for i in range(0,number_of_consumers-1):
            
            tau = i * 3600/6
            consumer = Consumer(f'Consumer {2+i}', A1, A2, Period1, Period2, Phi1, Phi2, offset, tau) 
            
            if i == 0:
                previous_supply_node = f'Node {i*4+2}'
                previous_return_node = f'Node {i*4+5}'
            else:
                previous_supply_node = f'Node {i*4+3}'
                previous_return_node = f'Node {i*4+6}'

            supply_node =  f'Node {i*4+7}'
            above_hex_node = f'Node {i*4+8}'
            under_hex_node = f'Node {i*4+9}'
            return_node = f'Node {i*4+10}'

            pipe_data = pipe_data_list[i+1]

            net.add_node(supply_node, 0, 0, 3*(i+2))
            net.add_node(above_hex_node, 2, 0, 3*(i+2))     
            net.add_node(under_hex_node, 2, 0, 3*(i+2)-1) 
            net.add_node(return_node, 1, 0, 3*(i+2)-1)

            net.add_pipe(f'Pipe {i*4 + 5}', previous_supply_node, supply_node, pipe_data) # needs to be connected to node from previous consumer
            net.add_pipe(f'Pipe {i*4 + 6}',supply_node,above_hex_node,pipe_data)
            net.add_pipe(f'Pipe {i*4 + 7}',under_hex_node,return_node,pipe_data)
            net.add_pipe(f'Pipe {i*4 + 8}',return_node,previous_return_node,pipe_data) # needs to be connected to node from previous consumer

            net.add_hex(f'Hex {i+2}', above_hex_node, under_hex_node, hex_data, pipe_data, consumer)

        # add overflow

        if use_overflow:
            overflow_node_supply = f'Node {i*4+11}'
            overflow_node_return = f'Node {i*4+12}'

            net.add_node(overflow_node_supply, 0, 0, 3*(i+2)+1)
            net.add_node(overflow_node_return, 1, 0, 3*(i+2)+1)

            net.add_pipe(f'Pipe {i*4 + 9}' , supply_node, overflow_node_supply, pipe_data)
            net.add_pipe(f'Pipe {i*4 + 10}', overflow_node_supply, overflow_node_return, pipe_data)
            net.add_pipe(f'Pipe {i*4 + 11}', overflow_node_return, return_node, pipe_data)

        return net
    
if __name__ == "__main__":

    # Test.model_network_Rutger()
    # Test.model_step_7()
    Test.test_network_builder()
    # Test.test_NR()
    # Test.test_incidence_and_loop_matrices()
