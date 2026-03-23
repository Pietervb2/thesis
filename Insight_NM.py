from baseclasses.pipe import Pipe

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def read_pipe_data(pipe_data_set):

    thesis_dir = os.path.dirname(os.path.abspath(__file__))
    constants_file = os.path.join(thesis_dir,'constants', 'constants_pipe.json')
    
    with open(constants_file) as f:
        constants = json.load(f)

    r_inner = constants[pipe_data_set]['r_inner'] # [m]
    r_outer = constants[pipe_data_set]['r_outer'] # [m]
    rho_pipe = constants[pipe_data_set]['rho_pipe'] # [kg / m3]
    rho_insu = constants[pipe_data_set]['rho_insu'] # [kg / m3]
    cp_pipe = constants[pipe_data_set]['cp_pipe']  # [J / kg K]
    cp_insu = constants[pipe_data_set]['cp_insu']  # [J/ kg K]
    k_pipe = constants[pipe_data_set]['k_pipe'] # thermal conductivity pipe [W / m K]
    k_insu = constants[pipe_data_set]['k_insu'] # thermal conductivity insulation [W / m K]
    h_pipe_air = constants[pipe_data_set]['h_pipe_air'] # natural convection from pipe to surrounding air [W / m2 K]
    insu_thickness = constants[pipe_data_set]['insu_thickness'] # [m]
    epsilon = constants[pipe_data_set]['epsilon'] # rougness of inner pipe [m]

    R = (
            np.log(r_outer/r_inner)/(2*np.pi*k_pipe) # conduction pipe
            + np.log((r_outer + insu_thickness)/r_outer)/(2*np.pi*k_insu) # conduction insulation
            + 1/(2*np.pi*(r_outer+insu_thickness)*h_pipe_air) # convection pipe to air
        )

    K = 1/R # total thermal conductivity [W / m K]

    pipe_data = [r_inner, r_outer, rho_pipe, rho_insu, cp_pipe, cp_insu, insu_thickness, K, epsilon]

    return pipe_data

# Create pipe
pipe_data = read_pipe_data('DN40')
pipe_length = 10
delta_z = 0
pipe1= Pipe('Pipe 1', pipe_length, delta_z, pipe_data)
pipe2 = Pipe('Pipe 2', pipe_length, delta_z, pipe_data)

dt = 1
num_steps = 100
T_init_water = 60
T_init_pipe = 60
v_inflow = 0.1
pipe1.bnode_init(dt, num_steps, T_init_water, T_init_pipe, v_inflow)
pipe2.bnode_init(dt, num_steps, T_init_water, T_init_pipe, v_inflow)

# Fill it with historic data
v_inflow = np.linspace(0.1,2,num_steps) 
mflow = v_inflow * np.pi * pipe1.r_inner**2 * pipe1.rho_water
mflow = np.concatenate([mflow])
# m_real = np.concatenate([mflow, np.zeros(num_steps), mflow[::-1]])
T_in = np.linspace(60,62,num_steps)
T_ambt = 20

fast = np.zeros(num_steps)
normal = np.zeros(num_steps)

for N in range(num_steps):
    pipe1.set_mflow(mflow[N],N) 
    pipe1.set_T_in(T_in[N],N)
    pipe1.bnode_method(T_ambt,N)
    pipe2.set_mflow(mflow[N],N)
    pipe2.set_T_in(T_in[N],N)  
    pipe2.bnode_method_fast(T_ambt,N)

plt.figure()
plt.plot(pipe1.T, label='Normal Method')
plt.plot(pipe2.T, label='Fast Method')
plt.legend()

# See what happens for different mflow
dis_num = 20 # number of different mass flow rates to test
T_out = np.zeros(dis_num)
T_stay = np.zeros(dis_num)
dis_mflow = np.linspace(0,3,dis_num) * np.pi * pipe2.r_inner**2 * pipe2.rho_water
for i,m_flow in enumerate(dis_mflow):
    pipe2.set_mflow(m_flow,N)
    pipe2.set_T_in(62,N)
    pipe2.bnode_method_fast(T_ambt,N)
    T_stay[i] = pipe2.t_stay_array[N]
    T_out[i] = pipe2.T[N]

    print(f'i = {i}, m_flow = {m_flow:.2f} kg/s, T_out = {T_out[i]:.2f} °C, t_stay = {T_stay[i]:.2f} s')


plt.figure()
plt.plot(dis_mflow, T_out, label='Fast Method with varying mflow')
plt.xlabel('Mass Flow [kg/s]')
plt.ylabel('Outlet Temperature [°C]')
plt.legend()


plt.figure()
plt.plot(dis_mflow, T_stay, label='Average stay time')
plt.xlabel('Mass Flow [kg/s]')
plt.ylabel('Average stay time [s]')
plt.grid()
plt.show()