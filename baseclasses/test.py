from network import Network 
from simulation import Simulation
from scipy.signal import square

import numpy as np
import matplotlib.pyplot as plt

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
                                                    plot_nodes_T = False, 
                                                    plot_pipes_T = False, 
                                                    plot_pipes_m_flow = False,
                                                    plot_network = False):

        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 1 # heat transmission coefficient DUMMY, NOTE de normale waarde is 0.4. Staat in het boek van Max p77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

        net = Network('one_loop')
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,10,0)
        net.add_node('Node 3',-10,10,0)
        net.add_node('Node 4',-10,20,0)
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

        dt = 1
        total_time = 500 # [s]

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
        sim.plot_node_temperature_results_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_results_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_results_network(net, v_flow, plot = plot_pipes_m_flow)            
       
        # plt.figure()

        # plt.plot(net.nodes['Node 1'].T - net.nodes['Node 2'].T, label = 'dT12')
        # plt.plot(net.nodes['Node 2'].T - net.nodes['Node 3'].T, label = 'dT23')
        # plt.plot(net.nodes['Node 3'].T - net.nodes['Node 4'].T, label = 'dT34')
        # plt.plot(net.nodes['Node 4'].T - net.nodes['Node 5'].T, label = 'dT45')
        # plt.plot(net.nodes['Node 5'].T - net.nodes['Node 6'].T, label = 'dT56')
        # plt.plot(net.nodes['Node 2'].T - net.nodes['Node 5'].T, label = 'dT25')

        # plt.legend()
        # plt.title('dT')
        # plt.grid(True)
            

        plt.show()

    def test_small_network_two_loops_full_simulation(temp_type, 
                                                     flow_type, 
                                                     plot_nodes_T = False, 
                                                     plot_pipes_T = False, 
                                                     plot_pipes_m_flow = False, 
                                                     plot_network = False):

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


        dt = 1
        total_time = 500 # [s]

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
        sim.plot_node_temperature_results_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_results_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_results_network(net, v_flow, plot = plot_pipes_m_flow)          

        plt.show()

    def test_pipe_four_nodes_simulation(temp_type,
                                        flow_type,
                                        plot_nodes_T = False, 
                                        plot_pipes_T = False, 
                                        plot_pipes_m_flow = False, 
                                        plot_network = False):
        
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

        dt = 1
        total_time = 500 # [s]

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
        sim.plot_node_temperature_results_network(net, T_inlet, plot = plot_nodes_T)
        sim.plot_pipe_temperature_results_network(net, T_inlet, plot = plot_pipes_T)
        sim.plot_pipe_m_flow_results_network(net, v_flow, plot = plot_pipes_m_flow)          

        plt.show()
    

if __name__ == "__main__":
    # Test.network_test_one_iteration()
    temp_type = "oscillation"
    temp_type2 = "constant"
    flow_type = "constant"
    Test.test_small_network_one_loop_full_simulation('constant', 'constant')
    # Test.test_small_network_two_loops_full_simulation('constant','constant')

    # Test.test_pipe_four_nodes_simulation('constant','constant', plot_nodes_T=True, plot_pipes_T = True)