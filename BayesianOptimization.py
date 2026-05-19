from datetime import datetime
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from test import optimization_run 

from scipy.special import logsumexp
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

        self.w_Tr = 8 * 1/40
        self.w_Ts = 11 * 1/60
        self.w_dTs = 5

        self.dict_w_Q = {
            'Profile 1': 0.1 * 1/60 * 1e-6,
            'Profile 2': 0.3 * 1/60 * 1e-6,
            'Profile 3': 1/60 * 1e-6,
            'Profile 4': 3 * 1/60 * 1e-6}
        
        self.dict_w_Qrespond = {
            'Profile 1': 4  * 1/60*1e-3,
            'Profile 2': 10 * 1/60*1e-3,
            'Profile 3': 30 * 1/60*1e-3,
            'Profile 4': 30 * 1/60*1e-3}
        
        # weight placed in order to make w_Q and w_Qrespond of the same importance
        
        # Saving cost function values and parameters for all iterations
        self.iter = 0
        self.dict_debug = {}

        self.dict_debug["w_Tr"] = self.w_Tr
        # self.dict_debug["w_Tof"] = self.w_Tof
        self.dict_debug["w_Ts"] = self.w_Ts
        self.dict_debug["dict_w_Q"] = self.dict_w_Q

        # self.dict_physical_bounds = {'Profile 1': {
        #                                 'theta_1': (60, 62.5),
        #                                 'theta_2': (62.5, 65),
        #                                 'theta_3': (200, 500e3),
        #                                 'theta_4': (0, 200e3),
        #                                 'theta_5': (30, 55),
        #                                 'theta_6': (1, 5)
        #                                 },
        #                             'Profile 2': {
        #                                 'theta_1': (60, 62.5),
        #                                 'theta_2': (62.5, 65),
        #                                 'theta_3': (0, 300e3),
        #                                 'theta_4': (0, 100e3),
        #                                 'theta_5': (30, 55),
        #                                 'theta_6': (1, 5)
        #                             }, 
        #                             'Profile 3': {
        #                                 'theta_1': (60, 62.5),
        #                                 'theta_2': (62.5, 65),
        #                                 'theta_3': (0, 300e3),
        #                                 'theta_4': (0, 200e3),
        #                                 'theta_5': (30, 55),
        #                                 'theta_6': (1, 5)
        #                             },
        #                             'Profile 4': {
        #                                 'theta_1': (60, 62.5),
        #                                 'theta_2': (62.5, 65),
        #                                 'theta_3': (0, 150e3),
        #                                 'theta_4': (0, 150e3),
        #                                 'theta_5': (30, 55),
        #                                 'theta_6': (1, 5)
        #                             }
        # }
        # # Physical bounds
        # self.PHYSICAL_BOUNDS = {
        #     'theta_1': (60, 65), 
        #     'theta_2': (0.1, 3),
        #     'theta_3': (0,10),
        #     'theta_4': (100e3, 400e3),
        #     'theta_5': (0, 55),
        #     'theta_6': (1, 3)
        # }
        self.dict_physical_bounds = {'Profile 1': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (200, 500e3),
                                            'theta_4': (0, 300e3),
                                            'theta_5': (30, 55)
                                            # 'theta_6': (1, 5)
                                            },
                                        'Profile 2': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (40, 350e3),
                                            'theta_4': (0, 200e3),
                                            'theta_5': (30, 55)
                                            # 'theta_6': (1, 5)
                                        }, 
                                        'Profile 3': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (0, 300e3),
                                            'theta_4': (0, 200e3),
                                            'theta_5': (30, 55)
                                            # 'theta_6': (1, 5)
                                        },
                                        'Profile 4': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (0, 150e3),
                                            'theta_4': (0, 150e3),
                                            'theta_5': (30, 55)
                                            # 'theta_6': (1, 5)
                                        }
            }
            # Physical bounds
        self.dict_debug["bounds"] = self.dict_physical_bounds[self.profile]

    def __call__(self, theta_1, theta_2, theta_3, theta_4, theta_5):
        """
        theta_1 : Minimum supply temperature [°C]
        theta_2 : Maximum supply temperature [°C]
        theta_3 : Heat demand threshold [W] (Q_set)
        theta_4 : Heat demand P-band [W]
        theta_5 : Temperature setpoint for overflow control [°C]
        theta_6 : Overflow valve P-band [°C]
        """

        # Denormalize all inputs
        theta_1 = denormalize(theta_1, *self.dict_physical_bounds[self.profile]['theta_1'])
        theta_2 = denormalize(theta_2, *self.dict_physical_bounds[self.profile]['theta_2'])
        theta_3 = denormalize(theta_3, *self.dict_physical_bounds[self.profile]['theta_3'])
        theta_4 = denormalize(theta_4, *self.dict_physical_bounds[self.profile]['theta_4'])
        theta_5 = denormalize(theta_5, *self.dict_physical_bounds[self.profile]['theta_5'])
        # theta_6 = denormalize(theta_6, *self.dict_physical_bounds[self.profile]['theta_6'])

        # print(f'theta_1: {theta_1}, theta_2: {theta_2}, theta_3: {theta_3}, theta_4: {theta_4}, theta_5: {theta_5}, theta_6: {theta_6}')

        # if theta_1 > theta_2:
        #     return -1e7  # Penalize invalid parameter combinations

        # Run simulation

        theta_6 = 3 # graden NOTE: voor nu even zo gedaan.

        net = optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5,theta_6,
                               self.profile,
                               self.dt,
                               self.pump_pressure,
                               self.curve,
                               run_type = self.run_type,
                               test_name = self.test_name)

        # Return temperature
        T_r = net.nodes['Node 1.6'].T
        mflow_r = net.pipes['Pipe 1.5']['pipe_instance'].mflow

        # Supply temperature
        T_s = net.nodes['Node 1.1'].T

        overflow = net.valves['Overflow valve'] 
        T_overflow = overflow.node.T

        # Heat demand and supply
        total_heat_demand = np.zeros_like(T_overflow)
        total_heat_supply = np.zeros_like(T_overflow)
        
        # Additional term to penalize slow response to changes in heat demand
        Q_respond = np.zeros(len(net.hexs))

        # Iterate through all heat exchangers and sum consumer demands
        i = 0
        for hex_obj in net.hexs.values():
            consumer = hex_obj.consumer
            total_heat_demand += consumer.Q_d 
            total_heat_supply += consumer.Q_supply


            two_min = int(120 / self.dt) # Number of timesteps in 2 minutes
            for idx in range(len(consumer.Q_d)-1-two_min):

                # smooth event trigger
                change_binary = (np.tanh(consumer.Q_d[idx+1] - consumer.Q_d[idx] - 100) + 1)/2  # indicating whether there is a change in heat demand, mapping it to zero or one, by subtracting -100 zero is really zero.     
                
                # how well it responded within 2 minutes based on how close the watts are. 
                Q_response = consumer.Q_d[idx+1:idx+1+two_min] - consumer.Q_supply[idx+1:idx+1+two_min]
                if self.dt == 60:
                    scaling_factor = 40
                elif self.dt == 1:
                    scaling_factor = 300

                if change_binary > 0.5:
                    pass
                Q_response_relu = softrelu(Q_response/scaling_factor)*scaling_factor # relu to only look at the positive difference
                
                Q_respond[i] += change_binary * np.sum(Q_response_relu) * self.dt
            
            i += 1

        warmup_period = 4.5 #h 
        length_of_simulation = len(T_overflow)
        warmup_steps = int(warmup_period / 24 * length_of_simulation)

        ### Return temperature term ###
        term_Tr = self.w_Tr * np.mean(T_r[warmup_steps:])**2 # Only looking at moments when there is no heat demand by applying exp(-Q_d/1e5) 

        ### Heat demand difference term ###      
        heat_demand_idx = [idx for idx,i in enumerate(total_heat_demand) if i>0] # Obtain all idx for total heat demand above zero 

        # Changed it to total heat demand per hex.
        term_Q = self.dict_w_Q[self.profile] * np.sum((total_heat_demand[heat_demand_idx] - total_heat_supply[heat_demand_idx])**2)*self.dt/23
        
        ### Heat demand waiting time response term ###
        term_Qrespond = self.dict_w_Qrespond[self.profile] * logsumexp(Q_respond) # So max delta_Q within those 2 minutes

        ### Supply term delivered ###
        term_Ts = self.w_Ts * np.mean(T_s[warmup_steps:])**2

        ### Change in supply temperature term ####
        dTs = np.diff(T_s[warmup_steps:]) 
        term_dTs = self.w_dTs * smooth_max(dTs**2)
     
        ### Regularization ###
        R = np.diag([1e-4, 1e-4, 1e-11, 1e-10, 1e-3, 1e-1]) # Regularization matrix 
        theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        regularization = theta.T @ R @ theta

        cost = term_Tr + term_Ts + term_dTs + term_Q + term_Qrespond + regularization

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
        print(f'term Tr {term_Tr}, term Ts {term_Ts}, term dTs {term_dTs}, term Q {term_Q}, term Qrespond {term_Qrespond}, regularization {regularization}, cost {cost}')
        # print(f'Regularization terms: {theta_1**2 * R[0,0]}, {theta_2**2 * R[1,1]}, {theta_3**2 * R[2,2]}, {theta_4**2 * R[3,3]}, {theta_5**2 * R[4,4]}, {theta_6**2 * R[5,5]}')
        return -cost

def denormalize(val, low, high):
    return low + val * (high - low)

def softrelu(x, alpha=1):
    return np.log(1 + np.exp(alpha * (x-2))) / alpha 

def smooth_max(x, alpha=100):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha

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
    pbounds = {k: (0, 1) for k in cost_fn.dict_physical_bounds[profile]}

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

    theta_1 = denormalize(optimizer.max['params']['theta_1'], *cost_fn.dict_physical_bounds[profile]['theta_1'])
    theta_2 = denormalize(optimizer.max['params']['theta_2'], *cost_fn.dict_physical_bounds[profile]['theta_2'])
    theta_3 = denormalize(optimizer.max['params']['theta_3'], *cost_fn.dict_physical_bounds[profile]['theta_3'])
    theta_4 = denormalize(optimizer.max['params']['theta_4'], *cost_fn.dict_physical_bounds[profile]['theta_4'])
    theta_5 = denormalize(optimizer.max['params']['theta_5'], *cost_fn.dict_physical_bounds[profile]['theta_5'])
    # theta_6 = denormalize(optimizer.max['params']['theta_6'], *cost_fn.dict_physical_bounds[profile]['theta_6'])

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
    theta_6 = 3 #NOTE
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