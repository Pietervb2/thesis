from baseclasses.test import optimization_run 

import numpy as np
import scipy.optimize
import time

def cost_function(theta):

    """
    theta_1 : Minimum supply temperature [°C]
    theta_2 : Maximum supply temperature [°C]
    theta_3 : Heat demand threshold [W] (Q_set)
    theta_4 : Heat demand P-band [W]
    theta_5 : Overflow valve temperature setpoint [°C]
    theta_6 : Overflow valve P-band [°C]
    """

    theta_1 = theta[0]
    theta_2 = theta[1]
    theta_3 = theta[2]
    theta_4 = theta[3]
    theta_5 = theta[4]
    theta_6 = theta[5]

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
    
    print(f'tot_cost = {cost}, theta_1 {theta_1}, theta_2 {theta_2}, theta_3 {theta_3}, theta_4 {theta_4}, theta_5 {theta_5}, theta_6 {theta_6}')
    # print(f'tot_cost = {cost}, term 1 {w_1 * np.sum(T_r**2)}, term 2 {w_2 * np.sum((T_set - T_overflow)**2)}, term 3 {w_3 * np.sum((total_heat_demand - total_heat_supply)**2)}, regularization {theta.T @ R @ theta}')
    # print(f'Regularization terms: {theta_1**2 * R[0,0]}, {theta_2**2 * R[1,1]}, {theta_3**2 * R[2,2]}, {theta_4**2 * R[3,3]}, {theta_5**2 * R[4,4]}, {theta_6**2 * R[5,5]}')
    return cost

profile = 'Profile 1'

# Initial guess
x0 = [60, 70, 250e3, 100e3, 55, 2]

bounds = [(60, 70), (60, 70), (0, 500e3), (0, 200e3), (54, 56), (0, 5)]  # e.g., pressures and flows must be non-negative


# Powell optimization

start = time.time()
result = scipy.optimize.minimize(cost_function, x0, method='Powell', bounds = bounds, options={'xtol':1e-3, 'ftol':1e-2, 'disp':True})
stop = time.time()  

print(f"Optimization time: {stop - start} seconds")
print(f'Success: {result.success}, Message: {result.message}')
print("Optimized variables:", result.x)
print("Objective function:", result.fun)

theta_1 = result.x[0]
theta_2 = result.x[1]
theta_3 = result.x[2]
theta_4 = result.x[3]
theta_5 = result.x[4]
theta_6 = result.x[5]

optimization_run(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, profile = profile, opt = False, opt_type = 'Powell')


# Nu vergelijk je dus wel de geparameteriseerde optie met BO terwijl je eigenlijk alle inputs op elke tijdstap apart zou kunnen optimaliseren.