"""
Replication of Simulation 1 from Gardner et al. (2014)
"Bayesian Optimization with Inequality Constraints"

Objective  (minimise):  l(x,y) = cos(2x)*cos(y) + sin(x)
Constraint (must be <=0.5): c(x,y) = cos(x)*cos(y) - sin(x)*sin(y)

Domain: x,y in [0, 6]

Implementation follows the same structure as BO_no_normalization_COP.py:
  - CostFunction class with run() / objective() / constraints() methods
  - BayesianOptimization + NonlinearConstraint from scipy
  - Matern kernel with per-parameter length-scale bounds scaled to physical range
  - Result plot replicating Figure 2 style from the paper
"""

from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import NonlinearConstraint

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


CONSTRAINT_LIMIT = 0.5


# ---------------------------------------------------------------------------
# Ground-truth functions
# ---------------------------------------------------------------------------

def objective_fn(x, y):
    return np.cos(2 * x) * np.cos(y) + np.sin(x)

def constraint_fn(x, y):
    return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)


# ---------------------------------------------------------------------------
# CostFunction class — same interface as BO_no_normalization_COP.py
# ---------------------------------------------------------------------------

class CostFunction:

    def __init__(self):
        self.dict_physical_bounds = {
            'x': (0.0, 6.0),
            'y': (0.0, 6.0),
        }
        self._cache = {}
        self.iter = 0
        self.dict_debug = {'bounds': self.dict_physical_bounds}

    def run(self, x, y):
        """Evaluates objective + constraint once and caches both."""
        key = (round(x, 6), round(y, 6))
        if key not in self._cache:
            cost = float(objective_fn(x, y))
            c    = float(constraint_fn(x, y))
            self._cache[key] = (cost, c)
            self.dict_debug[f'iter {self.iter}'] = {
                'x': x, 'y': y,
                'cost': cost, 'constraint': c,
                'feasible': c <= CONSTRAINT_LIMIT,
            }
            self.iter += 1
        return self._cache[key]

    def objective(self, x, y):
        """bayes_opt maximises — negate to minimise l(x,y)."""
        cost, _ = self.run(x, y)
        return -cost

    def constraints(self, x, y):
        """
        Returns a scalar float: c(x,y) - 0.5.
        bayes_opt 3.x expects a plain float for a single constraint.
        Must be <= 0 to be feasible.
        """
        _, c = self.run(x, y)
        return float(c - CONSTRAINT_LIMIT)


# ---------------------------------------------------------------------------
# run_bo — mirrors run_bo() in BO_no_normalization_COP.py
# ---------------------------------------------------------------------------

def run_bo():
    start = time.time()

    random_state = 1
    n_restarts   = 15
    alpha        = 1e-6
    init_points  = 5
    n_iter       = 24   # 30 total evaluations as in the paper (1 probe + 5 init + 24 iter)

    cost_fn = CostFunction()
    pbounds = cost_fn.dict_physical_bounds  # physical bounds directly, no normalisation

    constraint = NonlinearConstraint(cost_fn.constraints, -np.inf, 0.0)

    optimizer = BayesianOptimization(
        f            = cost_fn.objective,
        constraint   = constraint,
        pbounds      = pbounds,
        verbose      = 2,
        random_state = random_state,
    )

    # Kernel: same ratio logic as BO_no_normalization_COP.py
    # (0.05, 3) on [0,1] → (0.05*range, 3*range) per parameter
    param_order  = list(pbounds.keys())
    param_ranges = np.array([pbounds[k][1] - pbounds[k][0] for k in param_order])
    ls_init      = 1.0 * param_ranges
    ls_bounds    = np.column_stack([0.05 * param_ranges,
                                     3.0  * param_ranges])

    kernel = Matern(nu=2.5,
                    length_scale=ls_init,
                    length_scale_bounds=ls_bounds)

    optimizer.set_gp_params(kernel=kernel,
                             n_restarts_optimizer=n_restarts,
                             alpha=alpha)

    gp_kernel_before = optimizer._gp.kernel

    # Initial probe — top-right corner is feasible per Figure 2
    initial_point = {'x': 5.5, 'y': 5.5}
    optimizer.probe(initial_point, lazy=False)

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    gp_kernel_after = optimizer._gp.kernel_

    x_opt = optimizer.max['params']['x']
    y_opt = optimizer.max['params']['y']
    l_opt = -optimizer.max['target']
    c_opt = constraint_fn(x_opt, y_opt)

    stop = time.time()
    print(f'\n{"="*55}')
    print(f'Optimum found:  x={x_opt:.4f}, y={y_opt:.4f}')
    print(f'Objective l(x,y) = {l_opt:.4f}  (minimised)')
    print(f'Constraint c(x,y) = {c_opt:.4f}  (limit = {CONSTRAINT_LIMIT})')
    print(f'Feasible: {c_opt <= CONSTRAINT_LIMIT}')
    print(f'GP kernel before: {gp_kernel_before}')
    print(f'GP kernel after:  {gp_kernel_after}')
    print(f'Total evaluations: {len(optimizer.res)}')
    print(f'Total time: {stop - start:.2f}s')
    print(f'{"="*55}')

    return optimizer, cost_fn


# ---------------------------------------------------------------------------
# Plotting — replicates Figure 2 style from the paper
# ---------------------------------------------------------------------------

def plot_results(optimizer, cost_fn, save_path='simulation1_results.png'):
    grid_n = 300
    xi = np.linspace(0, 6, grid_n)
    yi = np.linspace(0, 6, grid_n)
    XX, YY = np.meshgrid(xi, yi)
    LL = objective_fn(XX, YY)
    CC = constraint_fn(XX, YY)
    feasible_mask = CC <= CONSTRAINT_LIMIT

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Objective contour
    cf = ax.contourf(XX, YY, LL, levels=20, cmap='RdYlBu_r', alpha=0.85)
    plt.colorbar(cf, ax=ax, label='l(x,y)')
    ax.contour(XX, YY, LL, levels=20, colors='k', linewidths=0.3, alpha=0.3)

    # Infeasible overlay (white semi-opaque) — matches paper Figure 2 style
    infeasible_rgba = np.zeros((*feasible_mask.shape, 4))
    infeasible_rgba[~feasible_mask] = [1, 1, 1, 0.65]
    ax.imshow(infeasible_rgba, origin='lower', extent=[0,6,0,6],
              aspect='auto', interpolation='none')

    # Constraint boundary
    ax.contour(XX, YY, CC, levels=[CONSTRAINT_LIMIT],
               colors='gray', linewidths=1.5, linestyles='--')

    # Scatter evaluated points
    feasible_pts   = []
    infeasible_pts = []
    for entry in optimizer.res:
        xv = entry['params']['x']
        yv = entry['params']['y']
        cv = constraint_fn(xv, yv)
        if cv <= CONSTRAINT_LIMIT:
            feasible_pts.append((xv, yv))
        else:
            infeasible_pts.append((xv, yv))

    if infeasible_pts:
        ix, iy = zip(*infeasible_pts)
        ax.scatter(ix, iy, marker='x', color='black', s=60, zorder=5,
                   linewidths=1.5, label=f'Infeasible ({len(infeasible_pts)})')

    if feasible_pts:
        fx, fy = zip(*feasible_pts)
        ax.scatter(fx, fy, marker='o', facecolors='white', edgecolors='black',
                   s=60, zorder=6, linewidths=1.5, label=f'Feasible ({len(feasible_pts)})')

    # Mark optimum
    x_opt = optimizer.max['params']['x']
    y_opt = optimizer.max['params']['y']
    ax.scatter([x_opt], [y_opt], marker='*', color='lime', s=280, zorder=7,
               edgecolors='black', linewidths=0.8, label='cBO optimum')

    ax.text(1.8, 3.2, 'INFEASIBLE', fontsize=13, color='gray',
            ha='center', va='center', fontweight='bold', alpha=0.7)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(
        'Constrained BO — Gardner (2014) Simulation 1\n'
        r'$\ell(x,y)=\cos(2x)\cos(y)+\sin(x)$,  '
        r'$c(x,y)\leq 0.5$',
        fontsize=11
    )
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved → {save_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    optimizer, cost_fn = run_bo()
    plot_results(optimizer, cost_fn, save_path='simulation1_results.png')
