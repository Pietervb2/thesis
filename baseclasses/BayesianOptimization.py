from bayes_opt import BayesianOptimization
from test import optimization_run 

import numpy as np

def cost_function(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6):

    """
    theta_1 : Minimum supply temperature [°C]
    theta_2 : Maximum supply temperature [°C]
    theta_3 : Heat demand threshold [W] (Q_set)
    theta_4 : Heat demand P-band [W]
    theta_5 : Overflow valve temperature setpoint [°C]
    theta_6 : Overflow valve P-band [°C]
    """

    if theta_1 > theta_2:
        return 1e7  # Penalize invalid parameter combinations

    # Run simulation
    net = optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, profile, opt = True)

    # Return temperature
    T_r = net.nodes['Node 1.6'].T

    # Overflow setpoint and temperature
    T_set = theta_5

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
    w_1 = 1e-2
    w_2 = 1
    w_3 = 1e-7

    R = np.diag([1e-3, 1e-3, 1e-10, 1e-10, 1e-3, 1e-1]) # Regularization matrix (identity for simplicity)
    theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
    
    # 𝑤_1 𝑇_𝑟^2+𝑤_2 (T_set−𝑇_𝑜verflow )^2+𝑤_3 (𝑄_(𝑑𝑒𝑚𝑎𝑛𝑑,𝑡𝑜𝑡)−𝑄_(𝑠𝑢𝑝,𝑡𝑜𝑡) )^2+𝜽R𝜽^T
    cost = w_1 * np.sum(T_r**2) + \
           w_2 * np.sum((T_set - T_overflow)**2) + \
           w_3 * np.sum((total_heat_demand - total_heat_supply)**2) + \
           theta.T @ R @ theta
    
    print(f'term 1 {w_1 * np.sum(T_r**2)}, term 2 {w_2 * np.sum((T_set - T_overflow)**2)}, term 3 {w_3 * np.sum((total_heat_demand - total_heat_supply)**2)}, regularization {theta.T @ R @ theta}')
    # print(f'Regularization terms: {theta_1**2 * R[0,0]}, {theta_2**2 * R[1,1]}, {theta_3**2 * R[2,2]}, {theta_4**2 * R[3,3]}, {theta_5**2 * R[4,4]}, {theta_6**2 * R[5,5]}')
    return -cost

# Bounded region of parameter space
pbounds = {'theta_1': (60, 65), 
           'theta_2': (65, 70), 
           'theta_3': (0, 500e3), 
           'theta_4': (0, 200e3), 
           'theta_5': (54, 56), 
           'theta_6': (0, 5)}

profile = 'Profile 1'

optimizer = BayesianOptimization(
    f=cost_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1)

optimizer.maximize(
    init_points=5,
    n_iter=10)

theta_1 =optimizer.max['params']['theta_1']
theta_2 =optimizer.max['params']['theta_2']
theta_3 =optimizer.max['params']['theta_3']
theta_4 =optimizer.max['params']['theta_4']
theta_5 =optimizer.max['params']['theta_5']
theta_6 =optimizer.max['params']['theta_6']

# theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = 60.88932373975674, 69.3343780526063, 160077.4634208071, 0.0, 55.46567498702314, 1.7212685648846227
optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, profile = profile, opt = False)