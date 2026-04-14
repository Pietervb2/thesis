from baseclasses.network import Network
from scipy.optimize import root

import pickle
import os
import numpy as np


# Load class instance at crash point
net_id = 'Profile 3'
Kvleak_bool = True
dis_steps = 125
c = 50
N = 27123
pump_type = 'curve'

if N is not None:
    file_name = f"{net_id}_N={N}_Kvleak={Kvleak_bool}_hsteps={dis_steps}_pump={c}kPa_{pump_type}.pkl"
else: 
    file_name = f"{net_id}_Kvleak={Kvleak_bool}_hsteps={dis_steps}_pump={c}kPa_{pump_type}.pkl"

    
# base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath('')

file = os.path.join(base_dir, 'debug', file_name)

with open(file, 'rb') as f:
    state = pickle.load(f)

net = Network.__new__(Network)  # create instance without calling __init__
net.__dict__.update(state)

# Determine N of the crashed state
zero_cols = (net.mflow_all == 0).all(axis=0)
non_zero_cols = ~zero_cols  # flip: True where column is NOT all-zero
idx = non_zero_cols[::-1].argmax()  # first non-zero from the right
N = len(non_zero_cols) - 1 - idx

pipe_ids = np.array(list(net.pipes.keys()))
p = len(pipe_ids)

# Initial guess for the mflows in the pipes
# mflow0 = net.mflow_all[:,N-1] 
# mflow0[mflow0 == 0] = 0.05 # Avoid zero initial guess for better convergence, can be tuned.
mflow0 = np.ones(p)*0.1

active_graph, active_mask1 = net.update_valves(N)
net.build_loop_matrix(active_graph)

# Reduce and create active incidence matrix
net.incidence_matrix_red = net.incidence_matrix[1:,:]
incidence_matrix_active = net.incidence_matrix_red[:, active_mask1]
net.incidence_matrix_active = incidence_matrix_active[~np.all(incidence_matrix_active == 0, axis=1), :]  # Remove zero rows (disconnected nodes)

# Adjust all vectors to active pipes
friction_vector = net.pressure_friction_vector + \
                net.Kp_array + \
                net.inv_Kv_array ** 2
                # np.round(net.inv_Kv_array ** 2)
net.friction_vector_active = friction_vector[active_mask1]

net.pressure_elevation_vector_active = net.pressure_elevation_vector[active_mask1]
net.pump_coeff_active = net.pump_coeff[active_mask1, :]
mflow0_active = mflow0[active_mask1]

result = root(net.res, mflow0_active, jac = net.jac, method = 'hybr', tol = 1e-5)
mflow = result.x

# Try it for different rounding of the vector
if result.success == False:

    for i in range(5,0,-1):          
        friction_vector = net.pressure_friction_vector + \
                        net.Kp_array + \
                        np.round(net.inv_Kv_array**2,i)
        net.friction_vector_active = friction_vector[active_mask1]
        
        # Solves system of non linear equations using scipy root function
        result = root(net.res, mflow0_active, jac = net.jac, method = 'hybr', tol = 1e-5)

        if result.success == True:
            print(f'iteration {i} was succesful')
            break
print(result.message)
print(f'residual sum of result.x {sum(net.res(mflow))}')

# print('mflow0:', mflow0_active)
print('N:', N)
print('friction:', sum(net.friction_vector_active))
print(f'pressure friction {sum(net.pressure_friction_vector)}')
print(f'Kp array {sum(net.Kp_array)}')
print('inv_Kv:', sum(net.inv_Kv_array))
print('loop_matrix shape:', net.loop_matrix_active.shape)
print('incidence shape:', net.incidence_matrix_active.shape)
# print(f' residual sum saved in net.mflow_all {np.sum(net.res(net.mflow_all[active_mask,N]))}')
# print(f' sum in net.mflow_all {sum(net.mflow_all[:,N])}')
# print(f'Are they equal {result.x == net.mflow_all[active_mask,N]}')
# print(result.x)
# print(f' mflow_all N active_mask {net.mflow_all[active_mask,N]}')