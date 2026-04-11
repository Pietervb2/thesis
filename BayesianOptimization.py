from datetime import datetime
from multiprocessing import Pool

from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern

import numpy as np
import time
import sys


from baseclasses.test import optimization_run 

# Physical bounds
PHYSICAL_BOUNDS = {
    'theta_1': (60, 65),
    'theta_2': (65, 70),
    'theta_3': (0, 500e3),
    'theta_4': (0, 200e3),
    'theta_5': (0, 3),
    'theta_6': (1, 5)
}

def run_bo(i):
    start = time.time()
    print(f'Initiation of code profile {i} at {datetime.now()}')
    pbounds = {k: (0, 1) for k in PHYSICAL_BOUNDS}
    profile = f'Profile {i}'

    # Settings 
    random_state = 1 
    n_restarts = 40
    alpha = 1e-6 
    init_points = 5
    n_iter = 10
    
    def denormalize(val, low, high):
        return low + val * (high - low)

    def cost_function(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6):

        """
        theta_1 : Minimum supply temperature [°C]
        theta_2 : Maximum supply temperature [°C]
        theta_3 : Heat demand threshold [W] (Q_set)
        theta_4 : Heat demand P-band [W]
        theta_5 : Additional temperature setpoint for overflow control [°C]
        theta_6 : Overflow valve P-band [°C]
        """

        # Denormalize all inputs
        theta_1 = denormalize(theta_1, *PHYSICAL_BOUNDS['theta_1'])
        theta_2 = denormalize(theta_2, *PHYSICAL_BOUNDS['theta_2'])
        theta_3 = denormalize(theta_3, *PHYSICAL_BOUNDS['theta_3'])
        theta_4 = denormalize(theta_4, *PHYSICAL_BOUNDS['theta_4'])
        theta_5 = denormalize(theta_5, *PHYSICAL_BOUNDS['theta_5'])
        theta_6 = denormalize(theta_6, *PHYSICAL_BOUNDS['theta_6'])

        # print(f'theta_1: {theta_1}, theta_2: {theta_2}, theta_3: {theta_3}, theta_4: {theta_4}, theta_5: {theta_5}, theta_6: {theta_6}')

        if theta_1 > theta_2:
            return -1e7  # Penalize invalid parameter combinations

        # Run simulation
        net = optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, profile, opt = True)

        # Return temperature
        T_r = net.nodes['Node 1.6'].T

        # Overflow setpoint and temperature
        T_set = 55 # [°C] Setpoint for overflow valve

        overflow = net.valves['Overflow valve'] 
        T_overflow = overflow.node.T

        # Heat demand and supply
        total_heat_demand = np.zeros_like(T_overflow)
        total_heat_supply = np.zeros_like(T_overflow)
        
        # Iterate through all heat exchangers and sum consumer demands
        for hex_obj in net.hexs.values():
            consumer = hex_obj.consumer
            total_heat_demand += consumer.Q_d
            total_heat_supply += consumer.Q_supply

        # Weights
        w_1 = 1e-1
        w_2 = 1e2
        w_3 = 0.2e-5

        R = np.diag([1e-4, 1e-4, 1e-11, 1e-10, 1e-3, 1e-1]) # Regularization matrix (identity for simplicity)
        theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        
        # 𝑤_1 𝑇_𝑟^2+𝑤_2 (T_set−𝑇_𝑜verflow )^2+𝑤_3 (𝑄_(𝑑𝑒𝑚𝑎𝑛𝑑,𝑡𝑜𝑡)−𝑄_(𝑠𝑢𝑝,𝑡𝑜𝑡) )^2+𝜽R𝜽^T

        warmup_period = 4.5 #h 
        length_of_simulation = len(T_overflow)
        warmup_steps = int(warmup_period / 24 * length_of_simulation)
        
        term1 = w_1 * np.mean(T_r[warmup_steps:]**2)
        term2 = w_2 * np.mean((T_set - T_overflow[warmup_steps:])**2)
        term3 = w_3 * np.mean((total_heat_demand - total_heat_supply)[warmup_steps:]**2)
        regularization = theta.T @ R @ theta

        cost = term1 + term2 + term3 + regularization
        
        print(f'term 1 {term1}, term 2 {term2}, term 3 {term3}, regularization {regularization}')
        # print(f'Regularization terms: {theta_1**2 * R[0,0]}, {theta_2**2 * R[1,1]}, {theta_3**2 * R[2,2]}, {theta_4**2 * R[3,3]}, {theta_5**2 * R[4,4]}, {theta_6**2 * R[5,5]}')
        return -cost

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=random_state)

    kernel = Matern(nu=1.5, 
                    length_scale=np.ones(optimizer.space.dim), 
                    length_scale_bounds=(0.05, 3))

    optimizer.set_gp_params(kernel=kernel, 
                            n_restarts_optimizer = n_restarts,
                            alpha = alpha)

    gp_kernel_before = optimizer._gp.kernel

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter)

    gp_kernel_after = optimizer._gp.kernel_

    theta_1 = denormalize(optimizer.max['params']['theta_1'], *PHYSICAL_BOUNDS['theta_1'])
    theta_2 = denormalize(optimizer.max['params']['theta_2'], *PHYSICAL_BOUNDS['theta_2'])
    theta_3 = denormalize(optimizer.max['params']['theta_3'], *PHYSICAL_BOUNDS['theta_3'])
    theta_4 = denormalize(optimizer.max['params']['theta_4'], *PHYSICAL_BOUNDS['theta_4'])
    theta_5 = denormalize(optimizer.max['params']['theta_5'], *PHYSICAL_BOUNDS['theta_5'])
    theta_6 = denormalize(optimizer.max['params']['theta_6'], *PHYSICAL_BOUNDS['theta_6'])

    print(f'GP kernel before optimization: {gp_kernel_before}'
        f'\nGP kernel after optimization: {gp_kernel_after}')


    results = {
        'params': optimizer.res,
        'max': optimizer.max,
        'kernel': str(optimizer._gp.kernel_),
        'bo_settings': {
            'init_points': init_points,
            'n_iter': n_iter,
            'n_restarts_optimizer': n_restarts,
            'random_state': random_state,
            'alpha' : alpha
        }
    }

    # Save best simulation result
    net = optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, profile, results, 
                           opt = False, 
                           opt_type = 'BO', 
                           n_init_points = init_points,
                           n_iter = n_iter)
    stop = time.time()
    print(f'Profile {i}: Total optimization time: {stop - start:.2f} seconds')

if __name__ == '__main__':    
    if len(sys.argv) > 1:
        profile_num = int(sys.argv[1])
        run_bo(profile_num)