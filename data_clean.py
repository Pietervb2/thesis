import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


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

def interpolate_irregular_data():
    # Read the irregular data

    val_dir =  os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(val_dir, 'data', 'pipe_validation', 'experiment')
    files = ['ExperimentA', 'ExperimentB', 'ExperimentC', 'ExperimentD']

    for file in files:
        file_loc = os.path.join(data_dir, f'{file}.csv')
        df = pd.read_csv(file_loc)
    
        # Create a regular time series with 1 second intervals
        time_min = int(df['Time'].min())
        time_max = int(df['Time'].max())
        regular_time = pd.Series(range(time_min, time_max + 1))
       
        # Create new dataframe with interpolated values
        new_df = pd.DataFrame({'Time': regular_time})     

        # Interpolate each column except time
        for column in df.columns:
            if column != ('Time' or 'MassFlowRate'):
                new_df[column] = np.round(np.interp(regular_time, df['Time'], df[column]),1)
        
        output_csv = os.path.join(data_dir, f'{file}_interpolated.csv')
        
        # The mass flow rate is constant over the whole experiment
        mass_flow_rate = df['MassFlowRate'][0]
        new_df['MassFlowRate'] = mass_flow_rate

        # Save to csv
        new_df.to_csv(output_csv, index=False)

def change_csv_file_layout_realdata():
    val_dir =  os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(val_dir, 'data', 'pipe_validation', 'experiment')
    files = ['ExperimentA', 'ExperimentB', 'ExperimentC', 'ExperimentD']

    for file in files:
        
        file_loc = os.path.join(data_dir, f'{file}_interpolated.csv')
        df_real = pd.read_csv(file_loc)

        # to Kelvin
        df_real[['OutletPipeTemp', 'OutletWaterTemp', 'InletPipeTemp', 'InletWaterTemp']] = np.round(df_real[['OutletPipeTemp', 'OutletWaterTemp', 'InletPipeTemp', 'InletWaterTemp']] + 273.15,1)
        
        # get rid of header
        df_real = df_real.iloc[1:]

        total_time = len(df_real)
        number_of_colums = df_real.shape[1]

        #Save as txt file with whitespace delimiter
        output_txt = os.path.join(data_dir, f'{file}_interpolated.txt')
        df_real.to_csv(output_txt, sep=' ', index=False, header = False)

        with open(output_txt, 'r') as csv:
            content = csv.read()

        with open(output_txt, 'w') as text_file:
            text_file.write(f"#1 \ndouble tab{total_time,number_of_colums} \n")  # Add header line
            text_file.write(content)

def clean_mo_csv(dt, T_ambt, file = None, temp_type = None, flow_type = None, length = None):
    """
    Function to put the modelica data into a 1 sec timestep
    """

    # Read the modelica data
    val_dir =  os.path.dirname(os.path.abspath(__file__))
    if file:
        mo_file = f'{file}_dt={dt}_Tambt={T_ambt}_mo.csv'
    else:
        mo_file = str(length) + 'm' + '_dt=' + str(dt) + '_Tin=' + temp_type + '_mflow=' + flow_type + '_Tambt=' + str(T_ambt) + '_mo.csv'
    df_modelica = pd.read_csv(os.path.join(val_dir,'data', 'pipe_validation','modelica', mo_file ), delimiter = ",")

    # Getting rid of interval values from modelica
    df_modelica['time'] = df_modelica['time'].astype(int)
    df_modelica = df_modelica.drop_duplicates(subset='time', keep='first') 

    # Save it
    clean_mo = mo_file.split('.')[0] + '_clean.csv'
    output_csv = os.path.join(val_dir,'data', 'pipe_validation','modelica', clean_mo)
    df_modelica.to_csv(output_csv, sep=',', index = False)

        
if __name__ == "__main__":
    T_ambt = 18
    files = ['A', 'B', 'C', 'D']
    dt_array = [1,1,1,30] # [s], delta time for every file
    for i in range(len(files)):
        temp_type = files[i]
        flow_type = files[i]    
        clean_mo_csv(dt_array[i],T_ambt,f'Experiment{files[i]}')    
    # # clean_mo_csv(length, 1, 'PipeDataULg151202','PipeDataULg151202',T_ambt)

    # length = 2000
    # dt = 30
    # clean_mo_csv(length, dt, 'constant','constant',T_ambt)
    # clean_mo_csv(length, dt, 'oscillation','constant',T_ambt)

    # clean_mo_csv(1,18,'ExperimentA')