from datetime import datetime
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import NonlinearConstraint
from bayes_opt.acquisition import ExpectedImprovement

from test import optimization_run, normal_run

import numpy as np
import time
import sys
import os
import pandas as pd


class CostFunction:
    def __init__(self, profile, dt, pump_pressure, curve, run_type, new_benchmark_run = False, test_name = None, split_Ts_bounds = True):
        self.profile = profile
        self.dt = dt
        self.pump_pressure = pump_pressure
        self.curve = curve

        self.run_type = run_type
        self.test_name = test_name
        self.split_Ts_bounds = split_Ts_bounds

        self.w_Tr = 1 * 1/4181 * 1e-3
        self.w_Ts = 1 * 1/4181 * 1e-3

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

        # Optionally widen theta_1/theta_2 to full overlapping range [60, 65]
        if not split_Ts_bounds:
            self.dict_physical_bounds[profile]['theta_1'] = (60, 65)
            self.dict_physical_bounds[profile]['theta_2'] = (60, 65)

        # theta_4 is fixed at the midpoint of its bounds
        bounds_4 = self.dict_physical_bounds[profile]['theta_4']
        self.theta_4_fixed = (bounds_4[0] + bounds_4[1]) / 2

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
        self.tol_Qrespond = 0.10
        self.tol_deltaQ = 0.10

        # Return temperature constraints
        self.T_r_max = 43

        # Saving values for constraint to avoid redundant simulations
        self._cache = {}

        # Debug purpose
        self.iter = 0
        self.dict_debug = {}
        self.dict_debug["w_Tr"] = self.w_Tr
        self.dict_debug["w_Ts"] = self.w_Ts
        self.dict_debug["split_Ts_bounds"] = split_Ts_bounds
        self.dict_debug["theta_4_fixed"] = self.theta_4_fixed
        self.dict_debug["bounds"] = self.dict_physical_bounds[self.profile]

    def run(self, theta_1, theta_2, theta_3, theta_5):
        """
        Runs simulation once and caches BOTH objective and constraint values.
        theta_4 is fixed at the midpoint of its bounds and not an optimization variable.

        theta_1 : Minimum supply temperature [°C]
        theta_2 : Maximum supply temperature [°C]
        theta_3 : Heat demand threshold [W] (Q_set)
        theta_5 : Temperature setpoint for bypass control [°C]
        """
        key = (round(theta_1, 4), round(theta_2, 4), round(theta_3, 4), round(theta_5, 4))

        if key not in self._cache:
            theta_4 = self.theta_4_fixed
            theta_6 = 3  # NOTE: hardcoded for now

            theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])

            net = optimization_run(theta,
                               self.profile,
                               self.dt,
                               self.pump_pressure,
                               self.curve,
                               run_type = self.run_type,
                               test_name = self.test_name)

            _, _, deltaQ_sq, Q_respond = self.compute_Q_terms(net)

            Q_respond_max = np.max(Q_respond)
            deltaQ_tot = np.sum(deltaQ_sq)

            T_r = net.nodes['Node 1.6'].T
            warmup_steps = int(4.5 / 24 * len(T_r))
            smooth_max_Tr = smooth_max(T_r[warmup_steps:])
            cost = self.compute_cost(net, theta)
            self._cache[key] = (cost, Q_respond_max, deltaQ_tot, smooth_max_Tr)

        return self._cache[key]

    def compute_cost(self, net, theta):

        if theta[0] > theta[1]:
            return 1e6

        T_r = net.nodes['Node 1.6'].T
        mflow_r = net.pipes['Pipe 1.6']['pipe_instance'].mflow

        # T_s = net.nodes['Node 1.1'].T
        T_s = net.T_in
        mflow_s = net.pipes['Pipe 1.1']['pipe_instance'].mflow

        warmup_period = 4.5  # h
        length_of_simulation = len(T_s)
        warmup_steps = int(warmup_period / 24 * length_of_simulation)

        # ------------------------------------------------------------------
        # Return temperature term
        # ------------------------------------------------------------------
        T_ATES_cold_well = 10
        c_p_water = 4186

        term_Tr = self.w_Tr * c_p_water * np.sum((T_r[warmup_steps:] - T_ATES_cold_well) * mflow_r[warmup_steps:]) * self.dt

        # ------------------------------------------------------------------
        # Supply temperature term
        # ------------------------------------------------------------------
        T_ATES_hot_well = 20
        term_Ts = self.w_Ts * c_p_water * np.sum((T_s[warmup_steps:] - T_ATES_hot_well) * mflow_s[warmup_steps:]) * self.dt

        # ------------------------------------------------------------------
        # Regularization
        # ------------------------------------------------------------------
        R = np.diag([1e-4, 1e-4, 1e-11, 1e-10, 1e-3, 1e-1])
        regularization = theta.T @ R @ theta

        cost = term_Tr + term_Ts + regularization

        self.dict_debug[f'iter {self.iter}'] = {
            'theta': theta.tolist(),
            'T_r': float(term_Tr),
            'T_s': float(term_Ts),
            'regularization': float(regularization),
            'cost': float(cost),
        }
        self.iter += 1

        return cost

    def compute_Q_terms(self, net):
        """
        Computes the heat demand terms for the optimization.
        """
        Q_respond = np.zeros(len(net.hexs))

        total_heat_demand = np.zeros(len(net.hexs))
        total_heat_supply = np.zeros(len(net.hexs))
        deltaQ_sq = np.zeros(len(net.hexs))

        for i, hex_obj in enumerate(net.hexs.values()):
            consumer = hex_obj.consumer
            total_heat_demand[i] = np.sum(consumer.Q_d)
            total_heat_supply[i] = np.sum(consumer.Q_supply)

            deltaQ_sq[i] = np.sum((consumer.Q_d - consumer.Q_supply)**2) * self.dt

            two_min = int(120 / self.dt)
            for idx in range(len(consumer.Q_d)-1-two_min):

                change_binary = (np.tanh(consumer.Q_d[idx+1] - consumer.Q_d[idx] - 100) + 1)/2

                Q_response = consumer.Q_d[idx+1:idx+1+two_min] - consumer.Q_supply[idx+1:idx+1+two_min]
                if self.dt == 60:
                    scaling_factor = 40
                elif self.dt == 1:
                    scaling_factor = 300

                Q_response_relu = softrelu(Q_response/scaling_factor)*scaling_factor
                Q_respond[i] += change_binary * np.sum(Q_response_relu) * self.dt

        return total_heat_demand, total_heat_supply, deltaQ_sq, Q_respond

    def objective(self, theta_1, theta_2, theta_3, theta_5):
        cost, *_ = self.run(theta_1, theta_2, theta_3, theta_5)
        return -cost  # bayes_opt maximizes

    def constraints(self, theta_1, theta_2, theta_3, theta_5):
        """Returns a vector of constraint values; each must be <= 0 to be feasible."""
        _, Q_respond_max, deltaQ_tot, smooth_max_Tr = self.run(
            theta_1, theta_2, theta_3, theta_5
        )

        print(f'Q_respond_max: {Q_respond_max - (1 + self.tol_Qrespond) * self.Q_respond_benchmark_max <= 0} ({Q_respond_max/self.Q_respond_benchmark_max}), deltaQ_tot: {deltaQ_tot - (1 + self.tol_deltaQ) * self.deltaQ_benchmark_tot <= 0} ({deltaQ_tot/self.deltaQ_benchmark_tot}), smooth_max_Tr: { smooth_max_Tr - self.T_r_max <= 0} ({smooth_max_Tr - self.T_r_max})')

        return np.array([
            Q_respond_max - (1 + self.tol_Qrespond) * self.Q_respond_benchmark_max,
            deltaQ_tot - (1 + self.tol_deltaQ) * self.deltaQ_benchmark_tot,
            smooth_max_Tr - self.T_r_max,
        ])


def softrelu(x, alpha=1):
    return np.log(1 + np.exp(alpha * (x-2))) / alpha

def smooth_max(x, alpha=100):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha

def make_constraint_gp(ls_init, ls_bounds, n_restarts=15, alpha=1e-6):
    """
    Factory for a constraint GP with a Matern(nu=2.5) kernel and per-parameter
    length-scale bounds scaled to the physical parameter ranges (same approach
    as the objective GP).

    Notes:
    - By default bayes_opt gives each constraint GP a scalar length_scale=1
      without bounds, so it cannot learn the correct spatial scale per
      parameter in physical (non-normalized) space.
    - normalize_y=True must be set explicitly here: bayes_opt's own GP has it
      by default, but we are constructing these GPs ourselves. Without it,
      the zero-mean prior would predict "constraint value ~ 0" (i.e. exactly
      on the feasibility boundary) in unexplored regions.
    - bayes_opt refits ALL constraint GPs automatically every iteration via
      AcquisitionFunction._fit_gp(), so replacing the models once before
      maximize() is sufficient; the length-scales are then re-fitted to the
      data at every iteration by maximizing the marginal likelihood.
    """
    return GaussianProcessRegressor(
        kernel=Matern(nu=2.5,
                      length_scale=ls_init.copy(),
                      length_scale_bounds=ls_bounds.copy()),
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
    )

def run_bo(i, dt, pump_pressure, curve, new_benchmark_run = False, split_Ts_bounds = True):
    start = time.time()
    print(f'Initiation of code profile {i} at {datetime.now()}')

    profile = f'Profile {i}'

    if new_benchmark_run == True:
        print("Perform benchmark simulation")
        T_in_benchmark = 65
        normal_run(profile, 'benchmark', dt, pump_pressure, curve, T_in_benchmark)

    # Settings
    random_state = 1
    n_restarts = 15
    alpha = 1e-6
    init_points = 10
    n_iter = 15

    cost_fn = CostFunction(profile, dt, pump_pressure, curve, run_type = 'optimization', split_Ts_bounds = split_Ts_bounds)

    # pbounds excludes theta_4 (fixed) and uses physical bounds directly (no normalization)
    pbounds = {k: v for k, v in cost_fn.dict_physical_bounds[profile].items() if k != 'theta_4'}

    constraint_lower = np.array([-np.inf, -np.inf, -np.inf])
    constraint_upper = np.array([0, 0, 0])

    constraint = NonlinearConstraint(cost_fn.constraints, constraint_lower, constraint_upper)

    optimizer = BayesianOptimization(
        f=cost_fn.objective,
        constraint = constraint,
        acquisition_function = ExpectedImprovement(xi = 15),
        pbounds=pbounds,
        verbose=2,
        random_state=random_state)

    # Scale lengthscales from normalized [0,1] range (0.05, 3) to physical ranges.
    # The ratio (0.05, 3) on [0,1] maps to (0.05 * range, 3 * range) per parameter.
    param_order = list(pbounds.keys())  # must match optimizer.space.dim order
    param_ranges = np.array([pbounds[k][1] - pbounds[k][0] for k in param_order])
    ls_init   = 1.0 * param_ranges                       # center: ls=1 in normalized -> range in physical
    ls_bounds = np.column_stack([0.05 * param_ranges,    # lower: 0.05 * range
                                  3.0  * param_ranges])   # upper: 3.0  * range

    kernel = Matern(nu=2.5,
                    length_scale=ls_init,
                    length_scale_bounds=ls_bounds)

    optimizer.set_gp_params(kernel=kernel,
                            n_restarts_optimizer = n_restarts,
                            alpha = alpha)

    # ------------------------------------------------------------------
    # Initialize constraint GPs
    # ------------------------------------------------------------------
    for idx in range(len(optimizer.constraint._model)):
        optimizer.constraint._model[idx] = make_constraint_gp(
            ls_init, ls_bounds,
            n_restarts=n_restarts,
            alpha=alpha,
        )

    gp_kernel_before = optimizer._gp.kernel
    constraint_kernels_before = [m.kernel for m in optimizer.constraint.model]

    bounds = cost_fn.dict_physical_bounds[profile]
    initial_point = {
        'theta_1': bounds['theta_1'][1],
        'theta_2': bounds['theta_2'][1],
        'theta_3': bounds['theta_3'][0],
        'theta_5': bounds['theta_5'][0] + 0.80 * (bounds['theta_5'][1] - bounds['theta_5'][0])
    }
    optimizer.probe(initial_point, lazy=False)

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter)

    gp_kernel_after = optimizer._gp.kernel_
    constraint_kernels_after = [m.kernel_ for m in optimizer.constraint.model]

    # Optimal parameters are already in physical space
    theta_1 = optimizer.max['params']['theta_1']
    theta_2 = optimizer.max['params']['theta_2']
    theta_3 = optimizer.max['params']['theta_3']
    theta_4 = cost_fn.theta_4_fixed
    theta_5 = optimizer.max['params']['theta_5']

    print(f'GP kernel before optimization: {gp_kernel_before}'
        f'\nGP kernel after optimization: {gp_kernel_after}')
    print(f'Constraint GP kernels before optimization: {constraint_kernels_before}'
        f'\nConstraint GP kernels after optimization: {constraint_kernels_after}')

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
        'theta_4_fixed': cost_fn.theta_4_fixed,
        'kernel': str(optimizer._gp.kernel_),
        'constraint_kernels': [str(m.kernel_) for m in optimizer.constraint.model],
        'bo_settings': {
            'init_points': init_points,
            'n_iter': n_iter,
            'n_restarts_optimizer': n_restarts,
            'random_state': random_state,
            'alpha' : alpha
        }
    }

    # Save best simulation result
    theta_6 = 3  # NOTE
    theta = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
    net = optimization_run(theta, profile, dt, pump_pressure, curve,
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
        # Optional second arg: 1/true/yes = split bounds (default), 0/false/no = unified bounds
        split = sys.argv[2].lower() not in ('0', 'false', 'no') if len(sys.argv) > 2 else True
        # print(f'split: {split}')
        dt = 1
        pump_pressure = 60
        curve = True
        run_bo(profile_num, dt, pump_pressure, curve, new_benchmark_run=False, split_Ts_bounds=split)