import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_pipe_data(file_path):
    # Read data
    data = pd.read_csv(file_path, delimiter=',', header=None)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot temperatures
    ax1.plot(data[0], data[2], label='Outlet Pipe Temp')
    ax1.plot(data[0], data[3], label='Outlet Water Temp')
    ax1.plot(data[0], data[4], label='Inlet Pipe Temp')
    ax1.plot(data[0], data[5], label='Inlet Water Temp')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot mass flow rate
    ax2.plot(data[0], data[1], label='Mass Flow Rate')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mass Flow Rate (kg/s)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def convert_mos_to_csv(): 

    root_path_mos = "C:\\Users\\piete\\Eneco\\Eneco - MasterThesis Pieter\\Simulatie\\Modelica\\modelica-ibpsa\\IBPSA\\Resources\\Data\\Fluid\\FixedResistances\\Validation\\PlugFlowPipes"
    files = ['PipeDataULg151202', 'PipeDataULg160118_1', 'PipeDataULg151204_4', 'PipeDataULg160104_2']

    root_dir = os.path.dirname(os.path.abspath(__file__))


    # Convert all files
    for file in files:
        full_path = os.path.join(root_path_mos, file + ".mos")
        save_path = os.path.join(root_dir, 'data', 'pipe_validation', file + ".csv")
        df = pd.read_csv(full_path, skiprows= [1], comment = "#", names=['Time', 'MassFlowRate', 'OutletPipeTemp', 
                                    'OutletWaterTemp', 'InletPipeTemp', 'InletWaterTemp'] )  
        df.to_csv(save_path, index = False)

def compare_node_method_and_modelica_no_pressure_drop(distances, timesteps, type_T, type_mflow):

    # NOTE: in both the node method and the modelica simulation there is no pressure drop

    for i in range(len(distances)):        
        val_dir =  os.path.dirname(os.path.abspath(__file__))
        mo_file = distances[i] + 'm' + '_dt=' + str(timesteps[i]) + '_Tin=' + type_T + '_mflow=' + type_mflow + '_mo.csv'

        data_modelica = pd.read_csv(os.path.join(val_dir, 'data', 'pipe_validation', mo_file))
        total_time = data_modelica['time'].iloc[-1]

        thesis_dir = os.path.dirname(val_dir)
        figure_folder_dir = 'network=One_pipe_#nodes_2_length=' + str(distances[i]) + '_dt=' + str(timesteps[i]) + '_total_time=' + str(total_time) + '_Tin=' + type_T + '_mflow=' + type_mflow
        node_file_path = os.path.join(thesis_dir, 'figures', figure_folder_dir, 'simulation_data.csv') 

        data_node = pd.read_csv(node_file_path)
    
        # extract temperature
        node_temp = data_node['T_Node 2']
        modelica_temp = data_modelica['T_sensor2.T']
        steady_state_diff = modelica_temp.iloc[-1] - node_temp.iloc[-1]
        ambient_temp = data_node['T_ambient']

        plots_folder = os.path.join(val_dir, 'figures', 'pipe_modelica', figure_folder_dir)

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)


        plt.figure()
        if type_T == 'constant' and type_mflow == 'constant':
            plt.title("Outlet temperature " + distances[i] + 'm'+ ', st.st diff= ' + str(round(steady_state_diff,5)))
        else:
            plt.title("Outlet temperature " + distances[i] + 'm')
            
        plt.plot(data_modelica['time'], modelica_temp, label = 'Modelica')
        plt.plot(data_node['time'], node_temp, label = 'Node Method')
        plt.plot(ambient_temp, label = 'Ambient Temperature')
        plt.plot(data_node['time'], data_node['T_inlet'], label = 'Inlet temperature', linestyle = '--')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.savefig(os.path.join(plots_folder, 'temperature_comparison.png'))

        plt.figure()
        plt.title("Input mass flow rate")
        plt.plot(data_node['time'], data_node['m_flow Pipe 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Mass Flow (kg/s)')
        plt.savefig(os.path.join(plots_folder, 'inlet_mass_flow.png'))

if __name__ == "__main__":
    # convert_mos_to_csv()
    distances = ['10', '50', '100', '1000']
    timesteps = ['1','10','10', '20']
    # compare_node_method_and_modelica_no_pressure_drop(distances, timesteps, 'constant', 'constant')
    compare_node_method_and_modelica_no_pressure_drop(['2000'], ['20'], 'oscillation', 'constant')