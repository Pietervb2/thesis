"""
costfunction_decompose_sweeps.py
=================================
Walks through every existing theta sweep in figures/sweeps/ and recomputes the
cost function for each swept value, broken down into its separate terms
(term_Tr, term_Ts, term_dTs, regularization), reusing the already-saved
per-value simulation output instead of rerunning the simulation.

Results are saved in the same format/location as the original sweep results:
figures/sweeps/cost_function_sweep_Profile_<p>_theta<t>/cost_function_terms_Profile_<p>_theta<t>.csv
"""

import datetime
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from BO import CostFunction

SWEEP_FOLDER_RE = re.compile(r'^cost_function_sweep_Profile_(\d+)_theta(\d+)$')
SIM_FOLDER_RE = re.compile(
    r'^cost_function_test_Profile_(\d+)_theta(\d+)_(.+)_dt=(\d+)_Tambt=(-?\d+)$'
)


class _FakeNode:
    def __init__(self, T):
        self.T = T


class _FakePipe:
    def __init__(self, mflow):
        self.mflow = mflow


def _load_simulation_arrays(sim_folder):
    node_temp = pd.read_csv(os.path.join(sim_folder, 'simulation_data', 'Node_temp.csv'))
    pipe_mflow = pd.read_csv(os.path.join(sim_folder, 'simulation_data', 'Pipe_mflow.csv'))
    theta = pd.read_csv(os.path.join(sim_folder, 'hex_consumer_data', 'theta.csv')).iloc[0].values

    # T_s = node_temp['Node 1.1'].values
    T_s = node_temp['T_in'].values
    T_r = node_temp['Node 1.6'].values
    mflow_s = pipe_mflow['Pipe 1.1'].values
    mflow_r = pipe_mflow['Pipe 1.6'].values

    return T_s, T_r, mflow_s, mflow_r, theta


def _build_fake_net(T_s, T_r, mflow_s, mflow_r):
    """Minimal stand-in exposing only what CostFunction.compute_cost reads from a Network."""
    return SimpleNamespace(
        nodes={'Node 1.1': _FakeNode(T_s), 'Node 1.6': _FakeNode(T_r)},
        pipes={
            'Pipe 1.1': {'pipe_instance': _FakePipe(mflow_s)},
            'Pipe 1.6': {'pipe_instance': _FakePipe(mflow_r)},
        },
    )


def find_sim_folders(sweeps_dir, profile_index, theta_index):
    """Every per-value simulation folder belonging to one sweep, sorted by swept value."""
    matches = []
    for name in os.listdir(sweeps_dir):
        m = SIM_FOLDER_RE.match(name)
        if not m:
            continue
        p, t, val, dt, _ = m.groups()
        if int(p) == profile_index and int(t) == theta_index:
            matches.append((float(val), int(dt), os.path.join(sweeps_dir, name)))
    matches.sort(key=lambda row: row[0])
    return matches


def process_sweep(sweeps_dir, profile_index, theta_index, pump_pressure=60, curve=True):
    sim_folders = find_sim_folders(sweeps_dir, profile_index, theta_index)
    if not sim_folders:
        print(f'  No simulation folders found, skipping.')
        return None

    profile = f'Profile {profile_index}'
    dt = sim_folders[0][1]
    cost_fn = CostFunction(profile, dt, pump_pressure, curve, run_type='test')

    rows = []
    for swept_val, _, sim_folder in sim_folders:
        try:
            T_s, T_r, mflow_s, mflow_r, theta = _load_simulation_arrays(sim_folder)
        except FileNotFoundError as exc:
            print(f'  Skipping {sim_folder}: {exc}')
            continue

        net = _build_fake_net(T_s, T_r, mflow_s, mflow_r)
        cost = cost_fn.compute_cost(net, theta)
        terms = cost_fn.dict_debug[f'iter {cost_fn.iter - 1}']

        cp = 4186.0
        eta = 0.9
        termTr_COP = float(np.sum(mflow_s * cp * (T_r - 10.0) ** 2 / (10.0 * eta)))
        termTs_COP = float(np.sum(mflow_s * cp * (T_s - 20.0) ** 2 / (eta * T_s)))

        rows.append((
            swept_val,
            terms['cost'],
            terms['T_r'],
            terms['T_s'],
            terms['dTs'],
            terms['regularization'],
            termTr_COP,
            termTs_COP,
        ))

    if not rows:
        return None

    return np.array(rows)


def save_results(sweep_folder, profile_index, theta_index, results):
    output_file = os.path.join(
        sweep_folder,
        f'cost_function_terms_Profile_{profile_index}_theta{theta_index}_Ts=Tin.csv',
    )
    np.savetxt(
        output_file,
        results,
        delimiter=',',
        header=f'theta_{theta_index},cost,term_Tr,term_Ts,term_dTs,regularization,termTr_COP,termTs_COP',
        comments='',
    )
    print(f'  Saved: {output_file}')


def main():
    start = datetime.datetime.now()
    print(f'Start: {start}')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sweeps_dir = os.path.join(base_dir, 'figures', 'sweeps')

    sweep_folders = sorted(
        name for name in os.listdir(sweeps_dir) if SWEEP_FOLDER_RE.match(name)
    )

    for name in sweep_folders:
        m = SWEEP_FOLDER_RE.match(name)
        profile_index, theta_index = int(m.group(1)), int(m.group(2))
        sweep_folder = os.path.join(sweeps_dir, name)

        print(f'Processing {name} ...')
        results = process_sweep(sweeps_dir, profile_index, theta_index)
        if results is not None:
            save_results(sweep_folder, profile_index, theta_index, results)

    print(f'Duration: {datetime.datetime.now() - start}')


if __name__ == '__main__':
    main()
