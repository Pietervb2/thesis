from network import Network 
from simulation import Simulation
from scipy.signal import square

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class Test:

    def network_test_one_iteration():
        # Pipe parameters
        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 0.4 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

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
        T_ambt = 20
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
                                                    plot_nodes_T = False, 
                                                    plot_pipes_T = False, 
                                                    plot_pipes_m_flow = False,
                                                    plot_network = False,
                                                    plot_nodes_dT = False):

        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 0.4 # heat transmission coefficient DUMMY, NOTE de normale waarde is 0.4. Staat in het boek van Max p77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

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

        T_ambt = 20 # [C]

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type)

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
                                                     plot_nodes_T = False, 
                                                     plot_pipes_T = False, 
                                                     plot_pipes_m_flow = False, 
                                                     plot_network = False,
                                                     plot_nodes_dT = False):

        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 1 # heat transmission coefficient DUMMY, de normale waarde is 0.4. Staat in het boek van Max p77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

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

        T_ambt = 20 # [C]

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type)

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
                                        plot_nodes_T = False, 
                                        plot_pipes_T = False, 
                                        plot_pipes_m_flow = False, 
                                        plot_network = False,
                                        plot_nodes_dT = False):
        
        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 1 # heat transmission coefficient DUMMY, NOTE de normale waarde is 0.4. Staat in het boek van Max p77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

        net = Network('pipe four nodes')
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,10,0)
        net.add_node('Node 3',0,20,0)
        net.add_node('Node 4',0,30,0)

        net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
        net.add_pipe('Pipe 2','Node 2','Node 3', pipe_data)	
        net.add_pipe('Pipe 3','Node 3','Node 4', pipe_data)	

        T_ambt = 20 # [C]

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type)

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
                                  T_ambt):
        
        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 1 # heat transmission coefficient DUMMY, NOTE de normale waarde is 0.4. Staat in het boek van Max p77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

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

            sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type)      

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
                    T_ambt,
                    plot_nodes_T = False, 
                    plot_pipes_T = False, 
                    plot_pipes_m_flow = False, 
                    plot_network = False,
                    plot_nodes_dT = False):
        
        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 0.4 # heat transmission coefficient DUMMY, NOTE de normale waarde is 0.4. Staat in het boek van Max p77. Is voor GEÏSOLEERDE pijpleiding!


        # k = 14.9 # [W/mK] thermal conductivity of the stainless steel
        # hi = 400 # [W/m^2K] convective heat transfer coefficient of air
        # ho = 6131.2 # [W/m^2K] convective heat transfer coefficient of water, from the 'Heat Transfer' of S. P. Venkateshan

        # K = 1/pipe_radius_outer * (np.log(pipe_radius_outer/pipe_radius_inner) / k + 1/(hi * pipe_radius_inner) + 1/(ho * pipe_radius_outer))**-1
        # print(K)

        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]
        pipe_length = total_length / (number_of_nodes -1)

        net = Network("One_pipe_#nodes_" + str(number_of_nodes) + "_length=" + str(total_length))

        for node in range(1, number_of_nodes + 1):
            net.add_node('Node ' + str(node), 0, (node - 1)*pipe_length, 0)
        
        for node in range(1, number_of_nodes):
            net.add_pipe('Pipe ' + str(node), 'Node ' + str(node), 'Node ' + str(node + 1), pipe_data) 
            print("Pipe "+str(node))

        sim = Simulation(dt, total_time, net.net_id, temp_type, flow_type)      

        if temp_type == "constant":
            T_inlet = np.ones(sim.num_steps) * 80                                 # Constant
        elif temp_type == "oscillation":
            T_inlet = 80 + 5 * np.sin(np.linspace(0, 8*np.pi, sim.num_steps))   # Oscillating inlet temperature
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


if __name__ == "__main__":

    # Test.network_test_one_iteration()

    # Test.test_small_network_one_loop_full_simulation('oscillation', 'constant', 20, 2000, plot_nodes_dT = True, plot_nodes_T = True, plot_pipes_T = True)
    # Test.test_small_network_two_loops_full_simulation('constant','constant')

    # number_of_nodes_array = [2,3,4,5,6,7,8,9,10]
    # Test.test_compare_bnode_method(number_of_nodes_array,'constant','constant',5,2000,2000)
    
    number_of_nodes = 2
    dt = 20
    total_time = 5000 # [s]
    total_length = 2000 # [m]
    T_ambt = 20 # [°C]
    Test.test_one_pipe(number_of_nodes, 'oscillation', 'constant', dt, total_time, total_length, T_ambt)

    # Test.test_pipe_four_nodes_simulation('constant','constant', plot_nodes_T=True, plot_pipes_T = True)

