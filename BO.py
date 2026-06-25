from datetime import datetime
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import NonlinearConstraint

from test import optimization_run, normal_run

import numpy as np
import time
import sys
import os
import pandas as pd


class CostFunction:
    def __init__(self, profile, dt, pump_pressure, curve, run_type, new_benchmark_run = False,test_name = None):
        self.profile = profile
        self.dt = dt
        self.pump_pressure = pump_pressure
        self.curve = curve

        self.run_type = run_type
        self.test_name = test_name

        self.w_Tr = 1 * 1/4181 * 1e-3
        self.w_Ts = 1 * 1/4181 * 1e-3
        self.w_dTs = 5

        # Physical bounds
        self.dict_physical_bounds = {'Profile 1': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (200e3, 500e3),
                                            'theta_4': (0, 300e3),
                                            'theta_5': (30, 55)
                                            },
                                        'Profile 2': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (40e3, 350e3),
                                            'theta_4': (0, 200e3),
                                            'theta_5': (30, 55)
                                        }, 
                                        'Profile 3': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (0, 300e3),
                                            'theta_4': (0, 200e3),
                                            'theta_5': (30, 55)
                                        },
                                        'Profile 4': {
                                            'theta_1': (60, 62.5),
                                            'theta_2': (62.5, 65),
                                            'theta_3': (0, 150e3),
                                            'theta_4': (0, 150e3),
                                            'theta_5': (30, 55)
                                        }
            }

        # Load heat constraints and tolerances        
        thesis_dir = os.path.dirname(os.path.abspath(__file__))
        file_loc = os.path.join(thesis_dir,
                                'figures',
                                'benchmark', 
                                f'Benchmark_{profile}_dt={dt}',
                                'hex_consumer_data',
                                'Q_hex.csv')
        Q_dp = pd.read_csv(file_loc, index_col=0)

        self.Q_respond_benchmark_max = Q_dp.loc['Max', 'Q_respond']
        self.deltaQ_benchmark_tot = Q_dp.loc['Total', 'DeltaQ_sq']

        print(f'Q_respond_benchmark = {self.Q_respond_benchmark_max}, deltaQ_benchmark = {self.deltaQ_benchmark_tot}')

        # Tolerances
        self.tol_Qrespond = 0.25
        self.tol_deltaQ = 0.25   # max allowed excess above deltaQ_tot benchmark

        # Return temperature constraints
        self.T_r_max = 43   # smooth-max of return temperature must stay below this [°C]

        # Saving values for constraint to avoid redundant simulations
        self._cache = {}

        # Debug purpose
        self.iter = 0
        self.dict_debug = {}
        self.dict_debug["w_Tr"] = self.w_Tr
        self.dict_debug["w_Ts"] = self.w_Ts
        self.dict_debug["bounds"] = self.dict_physical_bounds[self.profile]
    
    def run(self, theta_1, theta_2, theta_3, theta_4, theta_5):      
        """
        Runs simulation once and caches BOTH objective and constraint values
                         
        theta_1 : Minimum supply temperature [°C]
        theta_2 : Maximum supply temperature [°C]
        theta_3 : Heat demand threshold [W] (Q_set)
        theta_4 : Heat demand P-band [W]
        theta_5 : Temperature setpoint for overflow control [°C]
        theta_6 : Overflow valve P-band [°C]
        
        """
        key = (round(theta_1,4), round(theta_2,4), round(theta_3,4),
            round(theta_4,4), round(theta_5,4))
        
        if key not in self._cache:
            
            # Denormalize all inputs
            theta_1 = denormalize(theta_1, *self.dict_physical_bounds[self.profile]['theta_1'])
            theta_2 = denormalize(theta_2, *self.dict_physical_bounds[self.profile]['theta_2'])
            theta_3 = denormalize(theta_3, *self.dict_physical_bounds[self.profile]['theta_3'])
            theta_4 = denormalize(theta_4, *self.dict_physical_bounds[self.profile]['theta_4'])
            theta_5 = denormalize(theta_5, *self.dict_physical_bounds[self.profile]['theta_5'])
            theta_6 = 3 # NOTE: hardcoded for now

            theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
        
            net = optimization_run(theta,
                               self.profile,
                               self.dt,
                               self.pump_pressure,
                               self.curve,
                               run_type = self.run_type,
                               test_name = self.test_name)
                   
            _, _, deltaQ_sq, Q_respond = self.compute_Q_terms(net)

            # Compute objective and constraint values
            Q_respond_max = np.max(Q_respond)
            deltaQ_tot = np.sum(deltaQ_sq)

            # smooth-max of return temperature
            T_r = net.nodes['Node 1.6'].T
            warmup_steps = int(4.5 / 24 * len(T_r))
            smooth_max_Tr = smooth_max(T_r[warmup_steps:])
            cost = self.compute_cost(net, theta)
            self._cache[key] = (cost, Q_respond_max, deltaQ_tot, smooth_max_Tr)
        
        return self._cache[key]

    def compute_cost(self, net, theta):

        if theta[0] > theta[1]: # If min supply temperature is higher than max supply temperature, return a very high cost to make it infeasible
            return 1e6
        
        # Return temperature
        T_r = net.nodes['Node 1.6'].T
        mflow_r = net.pipes['Pipe 1.6']['pipe_instance'].mflow

        # Supply temperature
        T_s = net.T_in
        mflow_s = net.pipes['Pipe 1.1']['pipe_instance'].mflow

        warmup_period = 4.5 #h 
        length_of_simulation = len(T_s)
        warmup_steps = int(warmup_period / 24 * length_of_simulation)

        # ------------------------------------------------------------------
        # Return temperature term
        # ------------------------------------------------------------------
        T_ATES_cold_well = 10 # °C, this is the temperature of the cold well of the ATES
        c_p_water = 4186 # J/kgK, specific heat capacity of water
        
        term_Tr = self.w_Tr * c_p_water * np.sum((T_r[warmup_steps:] - T_ATES_cold_well) * mflow_r[warmup_steps:]) * self.dt

        # ------------------------------------------------------------------
        # Supply temperature term
        # ------------------------------------------------------------------       
        T_ATES_hot_well = 20 # °C, this is the temperature of the hot well of the ATES
        term_Ts = self.w_Ts * c_p_water * np.sum((T_s[warmup_steps:] - T_ATES_hot_well) * mflow_s[warmup_steps:]) * self.dt # Only penalize when there is actually supply, otherwise it can be noisy when there is no flow.

        # ------------------------------------------------------------------
        #  Change in supply temperature term
        # ------------------------------------------------------------------
        dTs = np.diff(T_s[warmup_steps:]) 
        term_dTs = self.w_dTs * smooth_max(dTs**2)

        # ------------------------------------------------------------------
        # Regularization
        # ------------------------------------------------------------------
        R = np.diag([1e-4, 1e-4, 1e-11, 1e-10, 1e-3, 1e-1]) # Regularization matrix 
        regularization = theta.T @ R @ theta

        cost = term_Tr + term_dTs + regularization

        # Save cost function values and parameters for debugging
        self.dict_debug[f'iter {self.iter}'] = {
            'theta': theta.tolist(),
            'T_r': float(term_Tr),
            'T_s': float(term_Ts),
            'dTs_niet_in_cost': float(term_dTs),
            'regularization': float(regularization),
            'cost' : float(cost),
        }
        self.iter += 1

        # print(f'term Tr {term_Tr}, term Ts {term_Ts}, term dTs {term_dTs}, regularization {regularization}, cost {cost}')
        # print(f'Regularization terms: {theta_1**2 * R[0,0]}, {theta_2**2 * R[1,1]}, {theta_3**2 * R[2,2]}, {theta_4**2 * R[3,3]}, {theta_5**2 * R[4,4]}, {theta_6**2 * R[5,5]}')
        return cost
    
    def compute_Q_terms(self, net):
        """
        Computes the heat demand terms for the optimization.
        """
        
        # Additional term to penalize slow response to changes in heat demand, this is ordered by hex
        Q_respond = np.zeros(len(net.hexs))

        # Heat demand and supply
        T_s = net.nodes['Node 1.1'].T
        total_heat_demand = np.zeros(len(net.hexs))
        total_heat_supply = np.zeros(len(net.hexs))
        deltaQ_sq = np.zeros(len(net.hexs))

        # Iterate through all heat exchangers and sum consumer demands
        for i, hex_obj in enumerate(net.hexs.values()):
            consumer = hex_obj.consumer
            total_heat_demand[i] = np.sum(consumer.Q_d) 
            total_heat_supply[i] = np.sum(consumer.Q_supply)

            deltaQ_sq[i] = np.sum((consumer.Q_d - consumer.Q_supply)**2) * self.dt

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

                Q_response_relu = softrelu(Q_response/scaling_factor)*scaling_factor # relu to only look at the positive difference               
                Q_respond[i] += change_binary * np.sum(Q_response_relu) * self.dt
                    
        return total_heat_demand, total_heat_supply, deltaQ_sq, Q_respond
    
    def objective(self, theta_1, theta_2, theta_3, theta_4, theta_5):
        cost, *_ = self.run(theta_1, theta_2, theta_3, theta_4, theta_5)
        return -cost  # bayes_opt maximizes

    def constraints(self, theta_1, theta_2, theta_3, theta_4, theta_5):
        """Returns a vector of constraint values; each must be <= 0 to be feasible."""
        _, Q_respond_max, deltaQ_tot, smooth_max_Tr = self.run(
            theta_1, theta_2, theta_3, theta_4, theta_5
        )

        print(f'Q_respond_max: {Q_respond_max - (1 + self.tol_Qrespond) * self.Q_respond_benchmark_max <= 0} ({Q_respond_max/self.Q_respond_benchmark_max}), deltaQ_tot: {deltaQ_tot - (1 + self.tol_deltaQ) * self.deltaQ_benchmark <= 0} ({deltaQ_tot/self.deltaQ_benchmark}), smooth_max_Tr: { smooth_max_Tr - self.T_r_max <= 0} ({smooth_max_Tr - self.T_r_max})')

        return np.array([
            # 1) Q_respond must not exceed (1 + tolerance) * benchmark
            Q_respond_max - (1 + self.tol_Qrespond) * self.Q_respond_benchmark_max,
            # 2) max (Q_d - Q_supply)^2 must not exceed (1 + epsilon) * benchmark
            deltaQ_tot - (1 + self.tol_deltaQ) * self.deltaQ_benchmark_tot,
            # 3) smooth-max of return temperature must stay below T_r_max
            smooth_max_Tr - self.T_r_max,
        ])
    
def denormalize(val, low, high):
    return low + val * (high - low)

def softrelu(x, alpha=1):
    return np.log(1 + np.exp(alpha * (x-2))) / alpha 

def smooth_max(x, alpha=100):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha

def run_bo(i, dt, pump_pressure, curve, new_benchmark_run = False):
    start = time.time()
    print(f'Initiation of code profile {i} at {datetime.now()}')
    
    profile = f'Profile {i}'

    if new_benchmark_run == True:
        print("Perform benchmark simulation")
        T_in_benchmark = 65 # °C, can be adjusted if needed
        normal_run(profile, 'benchmark', dt, pump_pressure, curve, T_in_benchmark)

    # Settings 
    random_state = 1 
    n_restarts = 15
    alpha = 1e-6
    init_points = 10
    n_iter = 15
   
    cost_fn = CostFunction(profile, dt, pump_pressure, curve, run_type = 'optimization')
    pbounds = {k: (0, 1) for k in cost_fn.dict_physical_bounds[profile]}

    constraint_lower = np.array([-np.inf, -np.inf, -np.inf])
    constraint_upper = np.array([0,0,0])

    constraint = NonlinearConstraint(cost_fn.constraints, constraint_lower, constraint_upper)

    optimizer = BayesianOptimization(
        f=cost_fn.objective,
        constraint = constraint,
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

    initial_point = {'theta_1': 1, 'theta_2': 1, 'theta_3': 0, 'theta_4': 0, 'theta_5': 0.80}
    optimizer.probe(initial_point, lazy=False)

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter)

    gp_kernel_after = optimizer._gp.kernel_

    theta_1 = denormalize(optimizer.max['params']['theta_1'], *cost_fn.dict_physical_bounds[profile]['theta_1'])
    theta_2 = denormalize(optimizer.max['params']['theta_2'], *cost_fn.dict_physical_bounds[profile]['theta_2'])
    theta_3 = denormalize(optimizer.max['params']['theta_3'], *cost_fn.dict_physical_bounds[profile]['theta_3'])
    theta_4 = denormalize(optimizer.max['params']['theta_4'], *cost_fn.dict_physical_bounds[profile]['theta_4'])
    theta_5 = denormalize(optimizer.max['params']['theta_5'], *cost_fn.dict_physical_bounds[profile]['theta_5'])

    print(f'GP kernel before optimization: {gp_kernel_before}'
        f'\nGP kernel after optimization: {gp_kernel_after}')

    results = {}
    for iter in range(len(optimizer.res)):
        results[f'iteration {iter}'] = {
            'theta': optimizer.res[iter]['params'],
            'target': optimizer.res[iter]['target'],
            'constraints' : optimizer.res[iter]['constraint'].tolist(),
            'allowed' : str(optimizer.res[iter]['allowed'])
        }

    results['final'] = {
        'max' : optimizer.max['params'],
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
                           debug_dict_BO = cost_fn.dict_debug)
    
    stop = time.time()
    print(f'Profile {i}: Total optimization time: {stop - start:.2f} seconds')

if __name__ == '__main__':    
    if len(sys.argv) > 1:
        profile_num = int(sys.argv[1])
        dt = 1
        pump_pressure = 60
        curve = True
        run_bo(profile_num, dt, pump_pressure, curve, new_benchmark_run = False)
