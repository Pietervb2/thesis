from bayes_opt import BayesianOptimization
from test import optimization_run 


def black_box_function(theta_1, theta_2, theta_3, theta_4):

    # Run simulation
    optimization_run(theta_1, theta_2, theta_3, theta_4)
    return 
