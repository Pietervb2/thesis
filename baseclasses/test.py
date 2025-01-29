from network import Network 
from simulation import Simulation
import numpy as np

class Test:

    def network_test_one_iteration():
        # Pipe parameters
        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 0.4 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

        net = Network()
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
        


    def network_test_full_simulation(plot_nodes_T = False, plot_pipes_T = False, plot_pipes_m_flow = False):
        pipe_radius_outer = 0.1 # [m] DUMMY
        pipe_radius_inner = 0.08 # [m] DUMMY
        K = 10 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
        pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

        net = Network()
        net.add_node('Node 1',0,0,0)
        net.add_node('Node 2',0,100,0)
        net.add_node('Node 3',-10,100,0)
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

        dt = 1
        total_time = 300 # [s]

        T_ambt = 20

        sim = Simulation(dt, total_time)
        
        v_flow = np.ones(sim.num_steps) * 2                           # Constant
        T_inlet = np.ones(sim.num_steps) * 80                          # Constant

        sim.simulate_network(net, T_inlet, v_flow, T_ambt)

        if plot_nodes_T:
            sim.plot_node_temperature_results_network(net, T_inlet)

        if plot_pipes_T:
            sim.plot_pipe_temperature_results_network(net, T_inlet)

        if plot_pipes_m_flow:
            sim.plot_pipe_m_flow_results_network(net, v_flow)


if __name__ == "__main__":
    # Test.network_test_one_iteration()
    Test.network_test_full_simulation(plot_nodes_T = True, plot_pipes_m_flow = False)


    ########################################################
    # De plot klopt niet dus kijke naar hoe de temperatuur door wordt gegeven. Maar het doorlopen van het netwerk werkt!!!!
    #######################################################