"""
deltaQ_Qrespond_decompose_sweeps.py
=================================
Walks through every existing theta sweep in figures/sweeps/ and extracts the
deltaQ and Q_respond constraint values used by BO.py's CostFunction.constraints
for each swept value, reusing the already-saved per-value simulation output
(hex_consumer_data/Q_hex.csv) instead of rerunning the simulation.

Results are saved in the same format/location as the original sweep results:
figures/sweeps/cost_function_sweep_Profile_<p>_theta<t>/deltaQ_Qrespond_Profile_<p>_theta<t>.csv
"""

import datetime
import os
import re

import numpy as np
import pandas as pd

from BO import CostFunction

SWEEP_FOLDER_RE = re.compile(r'^cost_function_sweep_Profile_(\d+)_theta(\d+)$')
SIM_FOLDER_RE = re.compile(
    r'^cost_function_test_Profile_(\d+)_theta(\d+)_(.+)_dt=(\d+)_Tambt=(-?\d+)$'
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


def _load_Q_terms(sim_folder):
    """deltaQ_tot and Q_respond_max exactly as used in CostFunction.run/constraints."""
    q_hex = pd.read_csv(os.path.join(sim_folder, 'hex_consumer_data', 'Q_hex.csv'), index_col=0)
    deltaQ_tot = q_hex.loc['Total', 'DeltaQ_sq']
    Q_respond_max = q_hex.loc['Max', 'Q_respond']
    return deltaQ_tot, Q_respond_max


def process_sweep(sweeps_dir, profile_index, theta_index, pump_pressure=60, curve=True):
    sim_folders = find_sim_folders(sweeps_dir, profile_index, theta_index)
    if not sim_folders:
        print('  No simulation folders found, skipping.')
        return None

    profile = f'Profile {profile_index}'
    dt = sim_folders[0][1]
    cost_fn = CostFunction(profile, dt, pump_pressure, curve, run_type='test')

    rows = []
    for swept_val, _, sim_folder in sim_folders:
        try:
            deltaQ_tot, Q_respond_max = _load_Q_terms(sim_folder)
        except FileNotFoundError as exc:
            print(f'  Skipping {sim_folder}: {exc}')
            continue

        constraint_deltaQ = deltaQ_tot - (1 + cost_fn.tol_deltaQ) * cost_fn.deltaQ_benchmark_tot
        constraint_Qrespond = Q_respond_max - (1 + cost_fn.tol_Qrespond) * cost_fn.Q_respond_benchmark_max

        rows.append((
            swept_val,
            deltaQ_tot,
            Q_respond_max,
            cost_fn.deltaQ_benchmark_tot,
            cost_fn.Q_respond_benchmark_max,
            constraint_deltaQ,
            constraint_Qrespond,
            float(constraint_deltaQ <= 0),
            float(constraint_Qrespond <= 0),
        ))

    if not rows:
        return None

    return np.array(rows)


def save_results(sweep_folder, profile_index, theta_index, results):
    output_file = os.path.join(
        sweep_folder,
        f'deltaQ_Qrespond_Profile_{profile_index}_theta{theta_index}.csv',
    )
    np.savetxt(
        output_file,
        results,
        delimiter=',',
        header=(
            f'theta_{theta_index},deltaQ_tot,Q_respond_max,deltaQ_benchmark_tot,Q_respond_benchmark_max,'
            'constraint_deltaQ,constraint_Qrespond,feasible_deltaQ,feasible_Qrespond'
        ),
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
