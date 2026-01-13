from network import Network 
from simulation import Simulation
from consumer import Consumer

from scipy.signal import square
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import root

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json


def test_compare_bnode_method(number_of_nodes_array,
                                temp_type,
                                flow_type,
                                dt,
                                total_time,
                                total_length,
                                pipe_data_set):
    
    pipe_data, total_length, T_ambt = read_pipe_data(pipe_data_set)
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

        T_in,v_flow = generate_input(temp_type, flow_type, total_time, dt)

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
        T_in, v_flow = generate_input(temp_type, flow_type, total_time, dt)
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
    simulate_network_compare(network,T_ambt, dt, 
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

    pipe_data = read_pipe_data('Pipe of experiment van der Heijden')
    hex_data = read_hex_data('Standard hex constants dummy pressure')

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
    
    T_in, v_flow = generate_input_one_pipe(temp_type,flow_type, total_time, dt)

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

    pipe_data = read_pipe_data('DN40')
    hex_data = read_hex_data('Standard hex constants dummy pressure')

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
    
    T_in, v_flow = generate_input_network(temp_type,flow_type, total_time, dt)

    # Run simulation
    sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type, flow_type = flow_type)      
    sim.simulate_network(net, T_in, v_flow, T_ambt, T_ambt, T_ambt)

def model_step_7():
    """
    Initial test to see whether the incidence and loop matrix construction is working properly.
    """
    pipe_data_DN40 = read_pipe_data('DN40')
    pipe_data_DN20 = read_pipe_data('DN20')
    hex_data = read_hex_data('Standard hex constants')
    pump_data = read_pump_data('20kPa Pump constant')
    
    # Create network
    net = Network("2 consumers without overflow normal, 20kPa")

    number_consumers = 2
    pipe_data_list = [pipe_data_DN40] * number_consumers
    h_initial_list = [0,0]

    heat_type1 = ['shower', 'shower']
    heat_type2 = ['shower']
    start_time1 = [8,18] #h
    start_time2 = [19] #h

    consumer1 = Consumer('Consumer 1',heat_type1, start_time1)
    consumer2 = Consumer('Consumer 2',heat_type2, start_time2)

    consumer_list = [consumer1, consumer2]

    network_builder(net, 
                    pipe_data_list, 
                    pipe_data_DN20, 
                    hex_data,
                    h_initial_list, 
                    pump_data, 
                    consumer_list, 
                    use_overflow = False)

    # Simulation parameters
    dt = 60 # s
    total_time = 24 * 3600 # s
    T_ambt = 20

    # Input profiles
    temp_type = 'constant'
    
    T_in = generate_input_network(temp_type, total_time, dt)

    # Run simulation
    sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)
    # sim.plot_network(net)      
    sim.simulate_network(net, T_in, T_ambt, T_ambt)

def model_network_Rutger():
    
    """
    Replicate network of which Rutger send data from
    """

    pipe_data_DN20 = read_pipe_data('DN20')
    pipe_data_DN25 = read_pipe_data('DN25')
    pipe_data_DN32 = read_pipe_data('DN32')
    pipe_data_DN40 = read_pipe_data('DN40')

    hex_data = read_hex_data('Standard hex constants')
    pump_data = read_pump_data('60kPa Pump curve')
    
    # Create network
    net = Network("Network Rutger actuating valves, Kvleak500")

    heat_demand_type1 = ['shower']
    heat_demand_type2 = ['shower']
    start_time1 = np.linspace(8,8.4,10)
    start_time2 = np.linspace(19,19.4,13) #h

    consumer_list = []
    for i in range(23):
        
        if i == 9:
            consumer = Consumer(f'Consumer {i+1}',heat_demand_type1, [start_time1[0]])
        else:
            consumer = Consumer(f'Consumer {i+1}',['nothing'], [start_time2[i-10]])
        consumer_list.append(consumer)

    pipe_data_list = [pipe_data_DN40] * 6 +[pipe_data_DN32] * 14 + [pipe_data_DN25] * 3
    # h_initial_list = np.linspace(0,1,len(consumer_list))
    h_initial_list = np.zeros(len(consumer_list)) # all closed
    
    network_builder(net, 
                    pipe_data_list,
                    pipe_data_DN20, 
                    hex_data,
                    h_initial_list, 
                    pump_data, 
                    consumer_list,
                    use_overflow = False)

    # Simulation parameters
    dt = 60 # s
    total_time = 24 * 3600 # h
    T_ambt = 20

    # Input profiles
    temp_type = 'constant'
    
    T_in = generate_input_network(temp_type, total_time, dt)

    # Run simulation
    sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)
    sim.simulate_network(net, T_in, T_ambt, T_ambt)

def test_incidence_and_loop_matrices():
    
    pipe_data = read_pipe_data('Pipe of DN40')
    hex_data = read_hex_data('Standard hex constants dummy pressure')

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

    pipe_data = read_pipe_data('DN40')
    pipe_data_hex = read_pipe_data('DN20')
    hex_data = read_hex_data('Standard hex constants')

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
    
    T_in = generate_input_network(temp_type, total_time, dt)

    # Run simulation
    sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)      
    sim.simulate_network(net, T_in, T_ambt, T_ambt)

def test_network_builder():

    pipe_data_DN40 = read_pipe_data('DN40')
    pipe_data_DN20 = read_pipe_data('DN20')
    hex_data = read_hex_data('Standard hex constants')
    pump_data = read_pump_data('50kPa Pump constant')
    
    # Create network
    net = Network("Network builder")
    number_consumers = 5
    pipe_data_list = [pipe_data_DN40] * number_consumers
    h_initial_list = [1] * number_consumers

    network_builder(net, 
                            pipe_data_list,
                            pipe_data_DN20, 
                            hex_data,
                            h_initial_list,
                            pump_data,
                            number_consumers)

    # Simulation parameters
    dt = 60 # s
    total_time = 24 * 3600 # h
    T_ambt = 20

    # Input profiles
    temp_type = 'constant'
    
    # Run simulation
    sim = Simulation(dt, total_time, net.net_id, T_ambt, temp_type = temp_type)
    sim.plot_network(net) 

def test_Rutger_data():

    # Try to find valve positions to replicate Rutgers data. 
    pipe_data_DN20 = read_pipe_data('DN20')
    pipe_data_DN25 = read_pipe_data('DN25')
    pipe_data_DN32 = read_pipe_data('DN32')
    pipe_data_DN40 = read_pipe_data('DN40')

    hex_data = read_hex_data('Standard hex constants')
    pump_data = read_pump_data('60kPa Pump constant')

    # Create network
    net = Network("Network Rutger")

    heat_type1 = 'shower'
    heat_type2 = 'shower'
    start_time1 = 8 #h
    start_time2 = 19 #h

    consumer_list = []
    for i in range(23):  
        if i <=9:
            consumer = Consumer(f'Consumer {i+1}',heat_type1, start_time1)
        else:
            consumer = Consumer(f'Consumer {i+1}',heat_type2, start_time2)
        consumer_list.append(consumer)

    pipe_data_list = [pipe_data_DN40] * 6 +[pipe_data_DN32] * 14 + [pipe_data_DN25] * 3 
    h_initial_list = [0]*len(consumer_list)

    network_builder(net, 
                    pipe_data_list,
                    pipe_data_DN20, 
                    hex_data,
                    h_initial_list, 
                    pump_data, 
                    consumer_list,
                    use_overflow = False)
    
    # Initialize with dummy variables
    net.initialize_network(1,1,np.array([65]),np.array([20]),np.array([20]))

    # Load data of 23 floor 'stijgstrangen'. 
    thesis_dir = os.path.dirname(os.path.abspath(__file__))
    average_v_file = os.path.join(os.path.dirname(thesis_dir),'data', 'data_23floors_Rutger.csv')
    average_v = pd.read_csv(average_v_file)['average v'].values

    # Use the diameters and velocities to fill the mflow array.
    mflow_array = np.zeros(1+23*6)

    for i in range(0,23):
        
        if i != 22:

            radius = net.pipes[f'Pipe {i+1}.1']['pipe_instance'].r_inner
            radius_next = net.pipes[f'Pipe {i+2}.1']['pipe_instance'].r_inner
            
            mflow = np.pi * radius ** 2 * average_v[i] * 1000
            mflow_next = np.pi * radius_next ** 2 * average_v[i+1] * 1000

            mflow_diff = mflow - mflow_next

            if i == 0:
                mflow_array[0] = mflow
                mflow_array[1] = mflow
                mflow_array[i+2:i+6] = mflow_diff
                mflow_array[6] = mflow
            else:
                mflow_array[i*6+1] = mflow
                mflow_array[i*6+2:i*6+6] = mflow_diff
                mflow_array[(i+1)*6] = mflow
            
        else:
            radius = net.pipes[f'Pipe {i+1}.1']['pipe_instance'].r_inner
            mflow = np.pi * radius ** 2 * average_v[i] * 1000
            mflow_array[i*6+1:(i+1)*6+1] = mflow
    print("mass flow array:", mflow_array)

    # Calculate pressure losses in loops without taking the valves into account
    friction_vector = net.pressure_friction_vector + \
                            net.Kp_array                            

    head_loss_without_valves  = net.loop_matrix @ (friction_vector * np.abs(mflow_array)*mflow_array)

    # Pump contribution (only in loop equations)
    net.pump_pressure_curve =  net.pump_coeff[:,0] * mflow_array**2 + \
                                        net.pump_coeff[:,1] * mflow_array + \
                                        net.pump_coeff[:,2]
        
    pump_term = net.loop_matrix @ net.pump_pressure_curve

    # Loop residual without valves
    loop_res = pump_term - head_loss_without_valves 
    
    # determine valve positions
    valve_positions = (net.inv_Kv_array != 0).astype(int)
    valve_flows_sq = net.loop_matrix @ (valve_positions * np.abs(mflow_array)*mflow_array)

    inv_Kv_squared = loop_res / valve_flows_sq
    Kv = np.sqrt(1 / inv_Kv_squared)
    print("Calculated Kv values for valves:",Kv)

    # Validate Kv values
    for i, hex_obj in enumerate(net.hexs.values()):        
        pipe_id, pipe_obj = next(iter(hex_obj.get_incoming_pipes().items()))
        j = net.pipe_map[pipe_id]
        net.inv_Kv_array[j] = 1/Kv[i]

    mflow0 = np.zeros(len(mflow_array))
    mflow0[:] = 0.1 
    result = root(net.res, mflow0, jac = net.jac, method = 'hybr')
    diff = np.sum(result.x - mflow_array)
    print(f"Difference between supplied Rutgers values and root solution: {diff}")
###########################################################
# Help functions for the tests
###########################################################

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
    Kp_dp = constants[hex_data_set]['Kp_dp'] # Pressure loss coefficient [-]
    K_vs = constants[hex_data_set]['K_vs'] # Hydrualic conductivity for valve [m3/s Pa^0.5]
    Kp = constants[hex_data_set]['Kp'] # Proportional gain coefficient for valve [-]
    Ki = constants[hex_data_set]['Ki'] # Integral gain coefficient for valve [-]

    # K_vleak = constants[hex_data_set]['K_vleak'] # Hydraulic conductivity for valve when closed[m3/s Pa^0.5]

    hex_data = [U, As, Kp_dp, K_vs, Kp, Ki]

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
                    pipe_data_list : list, 
                    pipe_hex_data, 
                    hex_data,
                    h_initial_list, 
                    pump_data,
                    consumer_list,
                    use_overflow = True):
        
        # Add heat exchangers and connecting pipes based on number of consumers

        number_of_consumers = len(consumer_list)


        net.add_node('Node 1.1', 0, 0, 0)
        net.add_node('Node 1.2', 0, 0, 3)
        net.add_node('Node 1.3', 2, 0, 3)
        net.add_node('Node 1.4', 2, 0, 2)
        net.add_node('Node 1.5', 1, 0, 2)
        net.add_node('Node 1.6', 1, 0, 0)

        net.add_pump('Pump 1', 'Node 1.6', 'Node 1.1', pipe_data_list[0], pump_data)

        net.add_pipe('Pipe 1.1','Node 1.1', 'Node 1.2', pipe_data_list[0])
        net.add_pipe('Pipe 1.2', 'Node 1.2', 'Node 1.3', pipe_data_list[0])

        net.add_hex('Hex 1', 'Node 1.3', 'Node 1.4', hex_data, pipe_hex_data, h_initial_list[0], consumer_list[0])

        net.add_pipe('Pipe 1.5', 'Node 1.4', 'Node 1.5', pipe_data_list[0])
        net.add_pipe('Pipe 1.6', 'Node 1.5', 'Node 1.6', pipe_data_list[0])
                
        
        for i in range(2,number_of_consumers+1):
            
            consumer = consumer_list[i-1]
            
            if i == 2:
                previous_supply_node = f'Node {i-1}.2'
                previous_return_node = f'Node {i-1}.5'
            else:
                previous_supply_node = f'Node {i-1}.1'
                previous_return_node = f'Node {i-1}.4'

            supply_node =  f'Node {i}.1'
            above_hex_node = f'Node {i}.2'
            under_hex_node = f'Node {i}.3'
            return_node = f'Node {i}.4'

            pipe_data = pipe_data_list[i-1]

            net.add_node(supply_node, 0, 0, 3*i)
            net.add_node(above_hex_node, 2, 0, 3*i)     
            net.add_node(under_hex_node, 2, 0, 3*i-1) 
            net.add_node(return_node, 1, 0, 3*i-1)

            net.add_pipe(f'Pipe {i}.1', previous_supply_node, supply_node, pipe_data) # needs to be connected to node from previous consumer
            net.add_pipe(f'Pipe {i}.2',supply_node,above_hex_node,pipe_data)
 
            net.add_hex(f'Hex {i}', 
                        above_hex_node, 
                        under_hex_node, 
                        hex_data, 
                        pipe_hex_data, 
                        h_initial_list[i-1],
                        consumer)
            
            net.add_pipe(f'Pipe {i}.5',under_hex_node,return_node,pipe_data)
            net.add_pipe(f'Pipe {i}.6',return_node,previous_return_node,pipe_data) # needs to be connected to node from previous consumer

        # add overflow

        if use_overflow:
            overflow_node_supply = f'Node {i}.7'
            overflow_node_return = f'Node {i}.8'

            net.add_node(overflow_node_supply, 0, 0, 3*i+1)
            net.add_node(overflow_node_return, 1, 0, 3*i+1)

            net.add_pipe(f'Pipe {i}.7' , supply_node, overflow_node_supply, pipe_data)
            net.add_pipe(f'Pipe {i}.8', overflow_node_supply, overflow_node_return, pipe_data)
            net.add_pipe(f'Pipe {i}.9', overflow_node_return, return_node, pipe_data)

        return net
    
if __name__ == "__main__":

    model_network_Rutger()
    # test_Rutger_data()
    # model_step_7()
