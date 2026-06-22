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


T_R_MAX = 43  # °C, must match BO.py CostFunction.T_r_max


def softrelu(x, alpha=1):
    return np.log(1 + np.exp(alpha * x)) / alpha


def smooth_max(x, alpha=100):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha


def _compute_smooth_max_Tr(sim_folder):
    node_temp_path = os.path.join(sim_folder, 'simulation_data', 'Node_temp.csv')
    df = pd.read_csv(node_temp_path, index_col=0)
    T_r = df['Node 1.6'].to_numpy()
    warmup_steps = int(4.5 / 24 * len(T_r))
    return smooth_max(T_r[warmup_steps:])


def _compute_Q_respond_1min_from_raw(sim_folder, dt):
    """Recompute Q_respond_1min per-hex from saved Q_d/Q_supply CSVs, matching simulation.py."""
    hex_dir = os.path.join(sim_folder, 'hex_consumer_data')
    hex_files = sorted(
        f for f in os.listdir(hex_dir) if re.match(r'^Hex \d+\.csv$', f)
    )
    two_min = int(120 / dt)
    one_min = int(60 / dt)
    scaling_factor = 40 if dt == 60 else 300
    Q_respond_1min_per_hex = []
    for fname in hex_files:
        df = pd.read_csv(os.path.join(hex_dir, fname))
        Q_d = df['Q_d'].to_numpy()
        Q_supply = df['Q_supply'].to_numpy()
        tot = np.zeros(len(Q_d))
        for idx in range(len(Q_d) - 1 - two_min):
            change_binary = (np.tanh(Q_d[idx + 1] - Q_d[idx] - 100) + 1) / 2



            Q_resp = Q_d[idx + 1:idx + 1 + one_min] - Q_supply[idx + 1:idx + 1 + one_min]
            tot[idx] = change_binary * np.sum(softrelu(Q_resp / scaling_factor) * scaling_factor) * dt

        Q_respond_1min_per_hex.append(np.sum(tot))
    return np.array(Q_respond_1min_per_hex)


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


def _load_Q_terms(sim_folder, dt):
    """deltaQ_tot, Q_respond_max, and Q_respond_1min stats as used in CostFunction."""
    q_hex = pd.read_csv(os.path.join(sim_folder, 'hex_consumer_data', 'Q_hex.csv'), index_col=0)
    deltaQ_tot = q_hex.loc['Total', 'DeltaQ_sq']
    Q_respond_max = q_hex.loc['Max', 'Q_respond']
    Q_respond_min = q_hex.loc['Min', 'Q_respond']
    Q_respond_mean = q_hex.loc['Mean', 'Q_respond']

    if 'Q_respond_1min' in q_hex.columns:
        Q_respond_1min_max = q_hex.loc['Max', 'Q_respond_1min']
        Q_respond_1min_min = q_hex.loc['Min', 'Q_respond_1min']
        Q_respond_1min_mean = q_hex.loc['Mean', 'Q_respond_1min']
    else:
        vals = _compute_Q_respond_1min_from_raw(sim_folder, dt)
        Q_respond_1min_max = vals.max()
        Q_respond_1min_min = vals.min()
        Q_respond_1min_mean = vals.mean()
    return deltaQ_tot, Q_respond_max, Q_respond_min, Q_respond_mean, Q_respond_1min_max, Q_respond_1min_min, Q_respond_1min_mean


def process_sweep(sweeps_dir, profile_index, theta_index, pump_pressure=60, curve=True):
    sim_folders = find_sim_folders(sweeps_dir, profile_index, theta_index)
    if not sim_folders:
        print('  No simulation folders found, skipping.')
        return None

    profile = f'Profile {profile_index}'
    dt = sim_folders[0][1]
    cost_fn = CostFunction(profile, dt, pump_pressure, curve, run_type='test')

    rows = []
    for swept_val, sim_dt, sim_folder in sim_folders:
        try:
            deltaQ_tot, Q_respond_max, Q_respond_min, Q_respond_mean, Q_respond_1min_max, Q_respond_1min_min, Q_respond_1min_mean = \
                _load_Q_terms(sim_folder, sim_dt)
            smooth_max_Tr = _compute_smooth_max_Tr(sim_folder)
        except FileNotFoundError as exc:
            print(f'  Skipping {sim_folder}: {exc}')
            continue

        constraint_deltaQ = deltaQ_tot - (1 + cost_fn.tol_deltaQ) * cost_fn.deltaQ_benchmark_tot
        constraint_Qrespond = Q_respond_max - (1 + cost_fn.tol_Qrespond) * cost_fn.Q_respond_benchmark_max
        constraint_Tr = smooth_max_Tr - T_R_MAX

        rows.append((
            swept_val,
            deltaQ_tot,
            Q_respond_max,
            Q_respond_min,
            Q_respond_mean,
            Q_respond_1min_max,
            Q_respond_1min_min,
            Q_respond_1min_mean,
            cost_fn.deltaQ_benchmark_tot,
            cost_fn.Q_respond_benchmark_max,
            constraint_deltaQ,
            constraint_Qrespond,
            float(constraint_deltaQ <= 0),
            float(constraint_Qrespond <= 0),
            smooth_max_Tr,
            constraint_Tr,
            float(constraint_Tr <= 0),
        ))

    if not rows:
        return None

    return np.array(rows)


def save_results(sweep_folder, profile_index, theta_index, results):
    output_file = os.path.join(
        sweep_folder,
        f'constraint_Profile_{profile_index}_theta{theta_index}.csv',
    )
    np.savetxt(
        output_file,
        results,
        delimiter=',',
        header=(
            f'theta_{theta_index},deltaQ_tot,Q_respond_max,Q_respond_min,Q_respond_mean,'
            'Q_respond_1min_max,Q_respond_1min_min,Q_respond_1min_mean,'
            'deltaQ_benchmark_tot,Q_respond_benchmark_max,'
            'constraint_deltaQ,constraint_Qrespond,feasible_deltaQ,feasible_Qrespond,'
            'smooth_max_Tr,constraint_Tr,feasible_Tr'
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
