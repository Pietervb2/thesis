from network import Network 
from simulation import Simulation
from scipy.signal import square

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json

class Test:

    def network_test_one_iteration(pipe_data_set):
        
        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        total_length = constants[pipe_data_set]['Length']
        T_ambt = constants[pipe_data_set]['T_ambt']

        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]

        net = Network('one loop one iteration')
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,5,0)
        net.add_node('Node 3',-10,5,0)
        net.add_node('Node 4',-10,10,0)
        net.add_node('Node 5',0,10,0)
        net.add_node('Node 6',0,100,0)
        net.add_node('Node 7',50,100,0)


        net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
        net.add_pipe('Pipe 2','Node 2','Node 3', pipe_data)	
        net.add_pipe('Pipe 3','Node 3','Node 4', pipe_data)	
        net.add_pipe('Pipe 4','Node 4','Node 5', pipe_data)	
        net.add_pipe('Pipe 5','Node 2','Node 5', pipe_data)	
        net.add_pipe('Pipe 6','Node 5','Node 6', pipe_data)	
        net.add_pipe('Pipe 7','Node 6','Node 7', pipe_data)	

        dt = 0.1
        num_steps = 100
        v_inflow = np.array([4]) 
        T_inlet = np.array([75.3])
        N = 0

        net.initialize_network(dt, num_steps, v_inflow, T_inlet)

        net.set_T_and_flow_network(T_ambt, v_inflow, T_inlet, N)

        for node_id, node in net.nodes.items():
            print(f'{node_id}, temperature : {node.T[0]}')
        
        for pipe_id in net.pipes.keys():
            pipe = net.pipes[pipe_id]['pipe_class']
            print(f'{pipe_id} -> pipe outlet temp : {pipe.T[0]}, pipe mass flow : {pipe.m_flow[0]}')        

    def test_small_network_one_loop_full_simulation(temp_type, 
                                                    flow_type,
                                                    dt, 
                                                    total_time,
                                                    pipe_data_set, 
                                                    plot_nodes_T = False, 
                                                    plot_pipes_T = False, 
                                                    plot_pipes_m_flow = False,
                                                    plot_network = False,
                                                    plot_nodes_dT = False):

        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        total_length = constants[pipe_data_set]['Length']
        T_ambt = constants[pipe_data_set]['T_ambt']

        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]

        net = Network('one_loop_2000')
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,10,0)
        net.add_node('Node 3',-2000,10,0)
        net.add_node('Node 4',-2000,20,0)
        net.add_node('Node 5',0,20,0)
        net.add_node('Node 6',0,30,0)
        net.add_node('Node 7',0,40,0)

        net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
        net.add_pipe('Pipe 2','Node 2','Node 3', pipe_data)	
        net.add_pipe('Pipe 3','Node 3','Node 4', pipe_data)	
        net.add_pipe('Pipe 4','Node 4','Node 5', pipe_data)	
        net.add_pipe('Pipe 5','Node 2','Node 5', pipe_data)	
        net.add_pipe('Pipe 6','Node 5','Node 6', pipe_data)	
        net.add_pipe('Pipe 7','Node 6','Node 7', pipe_data)	

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type, T_ambt)

        if temp_type == "constant":
            T_inlet = np.ones(sim.num_steps) * 80                                 # Constant
        elif temp_type == "oscillation":
            T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, sim.num_steps))   # Oscillating inlet temperature
        elif temp_type == "square":
            T_inlet = 80 + 1* square(2 * np.pi * sim.time / 20)                 # Square wave with a period of 20 steps
        
        if flow_type == "constant":
            v_flow = np.ones(sim.num_steps) * 2                           # Constant
        elif flow_type == "oscillation":
            v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, sim.num_steps)) # Oscillating flow velocity
        elif flow_type == "square":
            v_flow = 1.5 + 0.5 * square(2 * np.pi * sim.time / 50)        # Square wave flow velocity, 50 is the period


        sim.simulate_network(net, T_inlet, v_flow, T_ambt)
        
        # Plot outcome and save figure        
        sim.plot_network(net, plot = plot_network)
        sim.plot_node_temperature_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_network(net, v_flow, plot = plot_pipes_m_flow)
        sim.plot_node_difference_temperature_network(net,plot = plot_nodes_dT)               
        sim.save_data(net, T_inlet, v_flow, T_ambt)

        plt.show()

    def test_small_network_two_loops_full_simulation(temp_type, 
                                                     flow_type,
                                                     dt,
                                                     total_time,
                                                     pipe_data_set, 
                                                     plot_nodes_T = False, 
                                                     plot_pipes_T = False, 
                                                     plot_pipes_m_flow = False, 
                                                     plot_network = False,
                                                     plot_nodes_dT = False):

        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        total_length = constants[pipe_data_set]['Length']
        T_ambt = constants[pipe_data_set]['T_ambt']

        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]

        net = Network('two_loops')
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,5,0)
        
        net.add_node('Node 3',-10,5,0)
        net.add_node('Node 4',-10,10,0)
        net.add_node('Node 5',0,10,0)
        
        net.add_node('Node 6',0,100,0)
        net.add_node("Node 7", -10, 100, 0)
        net.add_node('Node 8', -10, 110, 0)
        net.add_node('Node 9', 0, 110, 0)

        net.add_node('Node 10',50,110,0)

        net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
        net.add_pipe('Pipe 2','Node 2','Node 3', pipe_data)	
        net.add_pipe('Pipe 3','Node 3','Node 4', pipe_data)	
        net.add_pipe('Pipe 4','Node 4','Node 5', pipe_data)	
        net.add_pipe('Pipe 5','Node 2','Node 5', pipe_data)	
        net.add_pipe('Pipe 6','Node 5','Node 6', pipe_data)	
        net.add_pipe('Pipe 7','Node 6','Node 7', pipe_data)	
        net.add_pipe('Pipe 8','Node 7','Node 8', pipe_data)
        net.add_pipe('Pipe 9','Node 8','Node 9', pipe_data)
        net.add_pipe('Pipe 10','Node 6','Node 9',pipe_data)
        net.add_pipe('Pipe 11','Node 9','Node 10',pipe_data)

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type, T_ambt)

        if temp_type == "constant":
            T_inlet = np.ones(sim.num_steps) * 80                                 # Constant
        elif temp_type == "oscillation":
            T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, sim.num_steps))   # Oscillating inlet temperature
        elif temp_type == "square":
            T_inlet = 80 + 1* square(2 * np.pi * sim.time / 20)                 # Square wave with a period of 20 steps
        
        if flow_type == "constant":
            v_flow = np.ones(sim.num_steps) * 2                           # Constant
        elif flow_type == "oscillation":
            v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, sim.num_steps)) # Oscillating flow velocity
        elif flow_type == "square":
            v_flow = 1.5 + 0.5 * square(2 * np.pi * sim.time / 50)        # Square wave flow velocity, 50 is the period

        # Plot outcome and save figure
        sim.simulate_network(net, T_inlet, v_flow, T_ambt)
        sim.plot_network(net, plot = plot_network)
        sim.plot_node_temperature_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_network(net, v_flow, plot = plot_pipes_m_flow)
        sim.plot_node_difference_temperature_network(net,plot = plot_nodes_dT)          
        sim.save_data(net, T_inlet, v_flow, T_ambt)
        plt.show()

    def test_pipe_four_nodes_simulation(temp_type,
                                        flow_type,
                                        dt,
                                        total_time,
                                        pipe_data_set,
                                        plot_nodes_T = False, 
                                        plot_pipes_T = False, 
                                        plot_pipes_m_flow = False, 
                                        plot_network = False,
                                        plot_nodes_dT = False):
        
        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        total_length = constants[pipe_data_set]['Length']
        T_ambt = constants[pipe_data_set]['T_ambt']
        
        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]

        net = Network('pipe four nodes')
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,10,0)
        net.add_node('Node 3',0,20,0)
        net.add_node('Node 4',0,30,0)

        net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
        net.add_pipe('Pipe 2','Node 2','Node 3', pipe_data)	
        net.add_pipe('Pipe 3','Node 3','Node 4', pipe_data)	

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type, T_ambt)

        if temp_type == "constant":
            T_inlet = np.ones(sim.num_steps) * 80                                 # Constant
        elif temp_type == "oscillation":
            T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, sim.num_steps))   # Oscillating inlet temperature
        elif temp_type == "square":
            T_inlet = 80 + 1* square(2 * np.pi * sim.time / 20)                 # Square wave with a period of 20 steps
        
        if flow_type == "constant":
            v_flow = np.ones(sim.num_steps) * 2                           # Constant
        elif flow_type == "oscillation":
            v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, sim.num_steps)) # Oscillating flow velocity
        elif flow_type == "square":
            v_flow = 1.5 + 0.5 * square(2 * np.pi * sim.time / 50)        # Square wave flow velocity, 50 is the period

    
        sim.simulate_network(net, T_inlet, v_flow, T_ambt)

        # Plot outcome and save figure
        sim.plot_network(net, plot = plot_network)
        sim.plot_node_temperature_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_network(net, v_flow, plot = plot_pipes_m_flow)
        sim.plot_node_difference_temperature_network(net,plot = plot_nodes_dT)
        sim.save_data(net, T_inlet, v_flow, T_ambt)           

        plt.show()

    def test_compare_bnode_method(number_of_nodes_array,
                                  temp_type,
                                  flow_type,
                                  dt,
                                  total_time,
                                  total_length,
                                  pipe_data_set):
        
        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        total_length = constants[pipe_data_set]['Length']
        T_ambt = constants[pipe_data_set]['T_ambt']

        
        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]
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

            if temp_type == "constant":
                T_inlet = np.ones(sim.num_steps) * 80                                 # Constant
            elif temp_type == "oscillation":
                T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, sim.num_steps))   # Oscillating inlet temperature
            elif temp_type == "square":
                T_inlet = 80 + 1* square(2 * np.pi * sim.time / 20)                 # Square wave with a period of 20 steps
            
            if flow_type == "constant":
                v_flow = np.ones(sim.num_steps) * 2                           # Constant
            elif flow_type == "oscillation":
                v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, sim.num_steps)) # Oscillating flow velocity
            elif flow_type == "square":
                v_flow = 1.5 + 0.5 * square(2 * np.pi * sim.time / 50)        # Square wave flow velocity, 50 is the period

            sim.simulate_network(net, T_inlet, v_flow, T_ambt)


            end_Node_T = net.nodes['Node ' + str(number_of_nodes)].T

            plt.plot(sim.time, end_Node_T, label = '#'+str(number_of_nodes - 2))        
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [°C]')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    ## TODO: make the plotting functions more logical and clearer setup. But that is for a later stadium. 

    def test_one_pipe(number_of_nodes,
                    temp_type,
                    flow_type,
                    dt,
                    total_time,
                    total_length,
                    pipe_data_set,
                    plot_nodes_T = False, 
                    plot_pipes_T = False, 
                    plot_pipes_m_flow = False, 
                    plot_network = False,
                    plot_nodes_dT = False):
        
        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        T_ambt = constants[pipe_data_set]['T_ambt']

        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]
        pipe_length = total_length / (number_of_nodes -1)

        net = Network("One_pipe_#nodes_" + str(number_of_nodes) + "_length=" + str(total_length))

        for node in range(1, number_of_nodes + 1):
            net.add_node('Node ' + str(node), 0, (node - 1)*pipe_length, 0)
        
        for node in range(1, number_of_nodes):
            net.add_pipe('Pipe ' + str(node), 'Node ' + str(node), 'Node ' + str(node + 1), pipe_data) 

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type, T_ambt)      

        if temp_type == "constant":
            T_inlet = np.ones(sim.num_steps) * 65                                 # Constant
        elif temp_type == "oscillation":
            T_inlet = 65 + 5 * np.sin(np.linspace(0, 8*np.pi, sim.num_steps))   # Oscillating inlet temperature
        elif temp_type == "square":
            T_inlet = 80 + 1* square(2 * np.pi * sim.time / 20)       

        if flow_type == "constant":
            v_flow = np.ones(sim.num_steps) * 2                           # Constant
        elif flow_type == "oscillation":
            v_flow = 1.5 + 0.8*np.cos(np.linspace(0, 2*np.pi, sim.num_steps)) # Oscillating flow velocity
        elif flow_type == "square":
            v_flow = 1.5 + 0.5 * square(2 * np.pi * sim.time / 50)        # Square wave flow velocity, 50 is the period

        sim.simulate_network(net, T_inlet, v_flow, T_ambt)

        # Plot outcome and save figure
        sim.plot_network(net, plot = plot_network)
        sim.plot_node_temperature_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_network(net, v_flow, plot = plot_pipes_m_flow)
        sim.plot_node_difference_temperature_network(net,plot = plot_nodes_dT)
        sim.save_data(net, T_inlet, v_flow, T_ambt)           

        plt.show()

    def test_real_input_data(pipe_data_set,
                        plot_nodes_T = False, 
                        plot_pipes_T = False, 
                        plot_pipes_m_flow = False, 
                        plot_network = False,
                        plot_nodes_dT = False):
        
        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(thesis_dir, 'constants.json')

        with open(constants_file) as f:
            constants = json.load(f)

        radius_outer = constants[pipe_data_set]['radius_outer']
        radius_inner = constants[pipe_data_set]['radius_inner'] 
        K = constants[pipe_data_set]['K']
        rho_pipe_mat = constants[pipe_data_set]['rho_pipe_mat']
        cp_pipe_mat = constants[pipe_data_set]['cp_pipe_mat']
        rho_insu = constants[pipe_data_set]['rho_insu']
        cp_insu = constants[pipe_data_set]['cp_insu']
        insu_thickness = constants[pipe_data_set]['insu_thickness']
        total_length = constants[pipe_data_set]['Length']
        T_ambt = constants[pipe_data_set]['T_ambt']

        pipe_data = [radius_outer, radius_inner, K, cp_pipe_mat, rho_pipe_mat, cp_insu, rho_insu, insu_thickness]

        number_of_nodes = 2
        pipe_length = total_length / (number_of_nodes -1)

        files = ['PipeDataULg151202', 'PipeDataULg160118_1', 'PipeDataULg151204_4', 'PipeDataULg160104_2']
        dt = [1,1,1,30] # [s], delta time for every file 


        for i, file in enumerate(files):

            net = Network("One_pipe_#nodes_" + str(number_of_nodes) + "_length=" + str(total_length))

            for node in range(1, number_of_nodes + 1):
                net.add_node('Node ' + str(node), 0, (node - 1)*pipe_length, 0)
            
            for node in range(1, number_of_nodes):
                net.add_pipe('Pipe ' + str(node), 'Node ' + str(node), 'Node ' + str(node + 1), pipe_data) 

            # For the real data
            basedir = 'c:/Users/piete/Eneco/Eneco - MasterThesis Pieter/Simulatie/thesis/validation'

            # Step through data to create smaller vectors at dt intervals

            T_inlet = pd.read_csv(os.path.join(basedir, 'data', 'pipe_validation', file + '_interpolated.csv'))['InletWaterTemp'].values
            m_flow = pd.read_csv(os.path.join(basedir, 'data', 'pipe_validation', file + '_interpolated.csv'))['MassFlowRate'].values

            total_time = len(T_inlet)

            T_inlet = T_inlet[::dt[i]]
            m_flow = m_flow[::dt[i]]

            pipe1 = net.pipes['Pipe 1']['pipe_class']
            v_flow = m_flow / pipe1.rho_water / pipe1.inner_cs       #TODO Temporary solution for now the mass flow data. Maybe later I should reconstruct the code 

            sim = Simulation(dt[i], total_time, net.net_id, file, file, T_ambt)      
            sim.simulate_network(net, T_inlet, v_flow, T_ambt)

            # Plot outcome and save figure
            sim.plot_network(net, plot = plot_network)
            sim.plot_node_temperature_network(net, T_inlet, plot = plot_nodes_T)
            sim.plot_pipe_temperature_network(net, T_inlet, plot = plot_pipes_T)
            sim.plot_pipe_m_flow_network(net, v_flow, plot = plot_pipes_m_flow)
            sim.plot_node_difference_temperature_network(net, plot = plot_nodes_dT)
            sim.save_data(net, T_inlet, v_flow, T_ambt)           

            plt.show()  

if __name__ == "__main__":


    number_of_nodes = 2
    dt = 30 # [s]
    total_time = 8000 # [s]
    total_length = 2000 # [m]

    # Test.test_one_pipe(number_of_nodes, 'constant', 'constant', dt, total_time, total_length, "Pipe of District Heating and Cooling Book", plot_nodes_T = True)

    Test.test_real_input_data("Pipe of experiment van der Heijden") # 20 T ambient
