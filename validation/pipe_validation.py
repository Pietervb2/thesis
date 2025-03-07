import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class PipeValidation:
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

    def compare_node_method_and_modelica_no_pressure_drop(distances, timesteps, type_T, type_mflow, T_ambt):

        # NOTE: in both the node method and the modelica simulation there is no pressure drop

        for i in range(len(distances)):        
            val_dir =  os.path.dirname(os.path.abspath(__file__))
            mo_file = distances[i] + 'm' + '_dt=' + str(timesteps[i]) + '_Tin=' + type_T + '_m_flow=' + type_mflow + '_Tambt=' + str(T_ambt) + '_mo.csv'

            data_modelica = pd.read_csv(os.path.join(val_dir, 'data', 'pipe_validation', mo_file))
            total_time = int(data_modelica['time'].iloc[-1])

            thesis_dir = os.path.dirname(val_dir)
            figure_folder_dir = 'network=One_pipe_#nodes_2_length=' + str(distances[i]) + '_dt=' + str(timesteps[i]) + '_total_time=' + str(total_time) + '_Tin=' + type_T + '_mflow=' + type_mflow + '_Tambt=' + str(T_ambt)         
            node_file_path = os.path.join(thesis_dir, 'figures', figure_folder_dir, 'simulation_data.csv') 

            data_node = pd.read_csv(node_file_path)
        
            # extract temperature
            node_temp = data_node['T_Node 2']
            modelica_temp = data_modelica['T_sensor2.T']
            steady_state_diff = modelica_temp.iloc[-1] - node_temp.iloc[-1]
            ambient_temp = data_node['T_ambient']

            plots_folder = os.path.join(val_dir, 'figures', 'node_modelica', figure_folder_dir)

            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)


            plt.figure()
            if type_T == 'constant' and type_mflow == 'constant':
                plt.title("Outlet temperature " + distances[i] + 'm'+ ', st.st diff= ' + str(round(steady_state_diff,5)))
            else:
                plt.title("Outlet temperature " + distances[i] + 'm')

            plt.plot(data_modelica['time'], modelica_temp, label = 'Modelica')
            plt.plot(data_node['time'], node_temp, label = 'Node Method')
            plt.plot(data_node['time'], ambient_temp, label = 'Ambient Temperature', linestyle = "--")
            plt.plot(data_node['time'], data_node['T_inlet'], label = 'Inlet Temperature', linestyle = '--')
            
            if not(type_T == 'constant' and type_mflow == 'constant'):
                plt.plot(data_modelica["time"], data_modelica["T_sensor1.T"], label = 'Inlet temperature modelica', linestyle = '--')
            
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

        
    def interpolate_irregular_data():
        # Read the irregular data

        val_dir =  os.path.dirname(os.path.abspath(__file__))
        files = ['PipeDataULg151202.csv', 'PipeDataULg160118_1.csv', 'PipeDataULg151204_4.csv', 'PipeDataULg160104_2.csv']

        for file in files:
            file_loc = os.path.join(val_dir, 'data', 'pipe_validation', file)
            df = pd.read_csv(file_loc)
        
            # Create a regular time series with 1 second intervals
            time_min = int(df['Time'].min())
            time_max = int(df['Time'].max())
            regular_time = pd.Series(range(time_min, time_max + 1))
            
            # Create new dataframe with interpolated values
            new_df = pd.DataFrame({'Time': regular_time})
            
            # Interpolate each column except time
            for column in df.columns:
                if column != 'Time':
                    new_df[column] = np.round(np.interp(regular_time, df['Time'], df[column]),1)
            
            output_csv = os.path.join(val_dir, 'data', 'pipe_validation', file.split('.')[0] + '_interpolated.csv')
            # Save to csv
            new_df.to_csv(output_csv, index=False)

    def compare_real_and_modelica_and_node_method_no_pressure_drop(T_ambt = 20):
        
        val_dir =  os.path.dirname(os.path.abspath(__file__))   
        files = ['PipeDataULg151202', 'PipeDataULg151204_4', 'PipeDataULg160118_1', 'PipeDataULg160104_2']
        dt = [1,1,1,30]
        thesis_dir = os.path.dirname(val_dir)

        for i, file in enumerate(files):

            # Read the real data
            file_loc = os.path.join(val_dir, 'data', 'pipe_validation', file + '_interpolated.csv')
            df_real = pd.read_csv(file_loc)
            
            # Read the node method data 
            figure_folder_dir = 'network=One_pipe_#nodes_2_length=' + '39' + '_dt=' + str(dt[i]) + '_total_time=' + str(len(df_real) - 1) + '_Tin=' + file + '_mflow=' + file + '_Tambt=' + str(T_ambt)
            file_loc_node = os.path.join(thesis_dir, 'figures', figure_folder_dir, 'simulation_data.csv')
            df_node = pd.read_csv(file_loc_node)

            # Read the modelica data
            mo_file = '39m' + '_dt=' + str(dt[i]) + '_Tin=' + file + '_mflow=' + file + '_Tambt=' + str(T_ambt) + '_mo.csv'
            df_modelica = pd.read_csv(os.path.join(val_dir, 'data', 'pipe_validation', mo_file))

            # Folder where the figure is saved
            plots_folder = os.path.join(val_dir, 'figures', 'real_modelica_node', figure_folder_dir)
            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)

            # Getting rid of interval values from modelica
            df_modelica['time'] = df_modelica['time'].astype(int)
            df_modelica = df_modelica.drop_duplicates(subset='time', keep='first')            
            modelica_temp = df_modelica['T_sensor2.T']
            
            plt.figure()
            plt.title("Temperature ")
            plt.plot(df_node["time"], df_real["OutletWaterTemp"][::dt[i]], label = 'Outlet Water Temp')
            plt.plot(df_node["time"], df_real["InletWaterTemp"][::dt[i]], label = 'Inlet Water Temp', linestyle = '--')
            plt.plot(df_node["time"], df_node["T_Node 2"], label = 'Node Method')
            plt.plot(df_node["time"], modelica_temp[::dt[i]], label = "Modelica")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, 'temperature_comparison.png'))

            plt.figure()
            plt.title("Pipe temperatures real data")
            plt.plot(df_real["Time"], df_real["OutletPipeTemp"], label = 'Outlet Pipe Temp')
            plt.plot(df_real["Time"], df_real["InletPipeTemp"], label = 'Inlet Pipe Temp')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, 'pipe_temperature'))


            plt.figure()
            plt.title("Mass Flow")
            plt.plot(df_real["Time"], df_real["MassFlowRate"])
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, 'mass_flow.png'))

    def change_csv_file_layout():
        val_dir =  os.path.dirname(os.path.abspath(__file__))   
        files = ['PipeDataULg151202', 'PipeDataULg160118_1', 'PipeDataULg151204_4' , 'PipeDataULg160104_2']

        for file in files:
            
            file_loc = os.path.join(val_dir, 'data', 'pipe_validation', file + '_interpolated.csv')
            df_real = pd.read_csv(file_loc)

            # to Kelvin
            df_real[['OutletPipeTemp', 'OutletWaterTemp', 'InletPipeTemp', 'InletWaterTemp']] = np.round(df_real[['OutletPipeTemp', 'OutletWaterTemp', 'InletPipeTemp', 'InletWaterTemp']] + 273.15,1)
            
            # get rid of header
            df_real = df_real.iloc[1:]

            total_time = len(df_real)
            number_of_colums = df_real.shape[1]

            #Save as txt file with whitespace delimiter
            output_txt = os.path.join(val_dir, 'data', 'pipe_validation', file + '_interpolated.txt')
            df_real.to_csv(output_txt, sep=' ', index=False, header = False)

            with open(output_txt, 'r') as csv:
                content = csv.read()

            with open(output_txt, 'w') as text_file:
                text_file.write(f"#1 \ndouble tab{total_time,number_of_colums} \n")  # Add header line
                text_file.write(content)

if __name__ == "__main__":
    # convert_mos_to_csv()
    distances = ['2000']
    timesteps = ['30']
    T_ambt = 20
    PipeValidation.compare_node_method_and_modelica_no_pressure_drop(distances, timesteps, 'oscillation', 'constant', T_ambt)
    # compare_research_and_node_method()
    # PipeValidation.compare_real_and_modelica_and_node_method_no_pressure_drop(T_ambt)