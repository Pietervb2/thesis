from datetime import datetime
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from test import optimization_run 

import numpy as np
import time
import sys

# T_supply[i] = theta_1 + theta_2 * np.tanh(theta_3 * (total_heat_demand[k] - theta_4)) # Alternative formulation with tanh function

class CostFunction:
    def __init__(self, profile, dt, pump_pressure, curve, run_type, test_name = None):
        self.profile = profile
        self.dt = dt
        self.pump_pressure = pump_pressure
        self.curve = curve

        self.run_type = run_type
        self.test_name = test_name

        self.w_Tr = 0.5
        self.w_Ts = 0.1
        self.w_dTs = 3

        self.dict_w_Q = {
            'Profile 1': 0.4e-7,
            'Profile 2': 0.4e-6,
            'Profile 3': 0.1e-5,
            'Profile 4': 0.1e-4}
        
        self.dict_w_Qrespond = {
            'Profile 1': 0.05e-3,
            'Profile 2': 0.08e-3,
            'Profile 3': 0.1e-3,
            'Profile 4': 0.1e-3
        }
        # Saving cost function values and parameters for all iterations
        self.iter = 0
        self.dict_debug = {}

        self.dict_debug["w_Tr"] = self.w_Tr
        # self.dict_debug["w_Tof"] = self.w_Tof
        self.dict_debug["w_Ts"] = self.w_Ts
        self.dict_debug["dict_w_Q"] = self.dict_w_Q

        # # Physical bounds
        # self.PHYSICAL_BOUNDS = {
        #     'theta_1': (60, 65), 
        #     'theta_2': (0.1, 3),
        #     'theta_3': (0,10),
        #     'theta_4': (100e3, 400e3),
        #     'theta_5': (0, 55),
        #     'theta_6': (1, 3)
        # }

        # Physical bounds
        self.PHYSICAL_BOUNDS = {
            'theta_1': (60, 65), 
            'theta_2': (65, 70),
            'theta_3': (0, 300e3),
            'theta_4': (0, 200e3),
            'theta_5': (0, 55),
            'theta_6': (1, 5)
        }
        
        self.dict_debug["bounds"] = self.PHYSICAL_BOUNDS

    def __call__(self, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6):
        """
        theta_1 : Minimum supply temperature [°C]
        theta_2 : Maximum supply temperature [°C]
        theta_3 : Heat demand threshold [W] (Q_set)
        theta_4 : Heat demand P-band [W]
        theta_5 : Temperature setpoint for overflow control [°C]
        theta_6 : Overflow valve P-band [°C]
        """

        # Denormalize all inputs
        theta_1 = denormalize(theta_1, *self.PHYSICAL_BOUNDS['theta_1'])
        theta_2 = denormalize(theta_2, *self.PHYSICAL_BOUNDS['theta_2'])
        theta_3 = denormalize(theta_3, *self.PHYSICAL_BOUNDS['theta_3'])
        theta_4 = denormalize(theta_4, *self.PHYSICAL_BOUNDS['theta_4'])
        theta_5 = denormalize(theta_5, *self.PHYSICAL_BOUNDS['theta_5'])
        theta_6 = denormalize(theta_6, *self.PHYSICAL_BOUNDS['theta_6'])

        # print(f'theta_1: {theta_1}, theta_2: {theta_2}, theta_3: {theta_3}, theta_4: {theta_4}, theta_5: {theta_5}, theta_6: {theta_6}')

        # if theta_1 > theta_2:
        #     return -1e7  # Penalize invalid parameter combinations

        # Run simulation
        net = optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6,
                               self.profile,
                               self.dt,
                               self.pump_pressure,
                               self.curve,
                               run_type = self.run_type,
                               test_name = self.test_name)

        # Return temperature
        T_r = net.nodes['Node 1.6'].T

        # Supply temperature
        T_s = net.nodes['Node 1.1'].T

        # Overflow setpoint and temperature
        T_set = 55 # [°C] Setpoint for overflow valve

        overflow = net.valves['Overflow valve'] 
        T_overflow = overflow.node.T

        # Heat demand and supply
        total_heat_demand = np.zeros_like(T_overflow)
        total_heat_supply = np.zeros_like(T_overflow)
        
        # Additional term to penalize slow response to changes in heat demand
        Q_respond = 0

        # Iterate through all heat exchangers and sum consumer demands
        for hex_obj in net.hexs.values():
            consumer = hex_obj.consumer
            total_heat_demand += consumer.Q_d
            total_heat_supply += consumer.Q_supply

            two_min = int(120 / self.dt) # Number of timesteps in 2 minutes
            for idx in range(len(consumer.Q_d)-1-two_min):

                Q_respond += (consumer.Q_d[idx+1] - consumer.Q_d[idx]) * softrelu((consumer.Q_d[idx+1:idx+1+two_min] - consumer.Q_supply[idx+1:idx+1+two_min])) 

        R = np.diag([1e-4, 1e-4, 1e-11, 1e-10, 1e-3, 1e-1]) # Regularization matrix 
        theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        
        warmup_period = 4.5 #h 
        length_of_simulation = len(T_overflow)
        warmup_steps = int(warmup_period / 24 * length_of_simulation)

        # Obtain all idx for total heat demand above zero 
        heat_demand_idx = [idx for idx,i in enumerate(total_heat_demand) if i>0]

        # Cost terms        
        term_Q = self.dict_w_Q[self.profile] * np.mean((total_heat_demand[heat_demand_idx] - total_heat_supply[heat_demand_idx])**2)

        term_Qrespond = self.dict_w_Qrespond[self.profile] * np.mean(Q_respond) # So averge delta_Q within those 2 minutes

        term_Tr = self.w_Tr * np.mean((np.exp(-total_heat_demand[warmup_steps:]/1e5)*T_r[warmup_steps:])**2)

        term_Ts = self.w_Ts * np.mean(T_s[warmup_steps:]**2)

        term_dTs = self.w_dTs * np.sum(np.diff(T_s[warmup_steps:])**2)

        regularization = theta.T @ R @ theta

        cost = term_Tr + term_Q + term_Ts + term_Qrespond + term_dTs + regularization

        # Save cost function values and parameters for debugging
        self.dict_debug[f'iter {self.iter}'] = {
            'theta': theta.tolist(),
            'T_r': float(term_Tr),
            'Q': float(term_Q),
            'T_s': float(term_Ts),
            'Qrespond': float(term_Qrespond),
            'dTs': float(term_dTs),
            'regularization': float(regularization),
            'cost' : float(cost)
        }
        self.iter += 1

        # term Tof_var {term_Tof_var}
        print(f'term Tr {term_Tr}, term Q {term_Q}, term Ts {term_Ts}, term Qrespond {term_Qrespond}, term dTs {term_dTs}, regularization {regularization}, cost {cost}')
        # print(f'Regularization terms: {theta_1**2 * R[0,0]}, {theta_2**2 * R[1,1]}, {theta_3**2 * R[2,2]}, {theta_4**2 * R[3,3]}, {theta_5**2 * R[4,4]}, {theta_6**2 * R[5,5]}')
        return -cost

def denormalize(val, low, high):
    return low + val * (high - low)

def softrelu(x, alpha=1):
    return np.log(1 + np.exp(alpha * x/500)) / alpha

def run_bo(i, dt, pump_pressure, curve):
    start = time.time()
    print(f'Initiation of code profile {i} at {datetime.now()}')
    
    profile = f'Profile {i}'

    # Settings 
    random_state = 1 
    n_restarts = 15
    alpha = 1e-6
    init_points = 10
    n_iter = 10
   
    cost_fn = CostFunction(profile, dt, pump_pressure, curve, run_type = 'optimization')
    pbounds = {k: (0, 1) for k in cost_fn.PHYSICAL_BOUNDS}

    optimizer = BayesianOptimization(
        f=cost_fn,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=random_state)

    kernel = Matern(nu=2.5, 
                    length_scale=np.ones(optimizer.space.dim), 
                    length_scale_bounds=(0.05, 3))

    optimizer.set_gp_params(kernel=kernel, 
                            n_restarts_optimizer = n_restarts,
                            alpha = alpha)
    
    # # Verschil met de alpha en de white Kernel is dat ze de white kernel ook afschatten tijdens set_gp_params
    # noise_level = 1e-3
    # WhiteKernel(noise_level = noise_level)

    gp_kernel_before = optimizer._gp.kernel

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter)

    gp_kernel_after = optimizer._gp.kernel_

    theta_1 = denormalize(optimizer.max['params']['theta_1'], *cost_fn.PHYSICAL_BOUNDS['theta_1'])
    theta_2 = denormalize(optimizer.max['params']['theta_2'], *cost_fn.PHYSICAL_BOUNDS['theta_2'])
    theta_3 = denormalize(optimizer.max['params']['theta_3'], *cost_fn.PHYSICAL_BOUNDS['theta_3'])
    theta_4 = denormalize(optimizer.max['params']['theta_4'], *cost_fn.PHYSICAL_BOUNDS['theta_4'])
    theta_5 = denormalize(optimizer.max['params']['theta_5'], *cost_fn.PHYSICAL_BOUNDS['theta_5'])
    theta_6 = denormalize(optimizer.max['params']['theta_6'], *cost_fn.PHYSICAL_BOUNDS['theta_6'])

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
    net = optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, profile, dt, pump_pressure, curve, 
                           run_type = 'save_optimization',
                           opt_results = results, 
                           n_init_points = init_points,
                           n_iter = n_iter,
                           new_benchmark_run = True,
                           debug_dict_BO = cost_fn.dict_debug)
    
    stop = time.time()
    print(f'Profile {i}: Total optimization time: {stop - start:.2f} seconds')

if __name__ == '__main__':    
    if len(sys.argv) > 1:
        profile_num = int(sys.argv[1])
        dt = 1
        pump_pressure = 60
        curve = True
        run_bo(profile_num, dt, pump_pressure, curve)