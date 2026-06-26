import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseclasses.consumer import Consumer


def consumer_start_times(profile, peaks):
    if profile == 'Profile 1':
        heat_demand_types = ['shower', 'shower']
        start_times = peaks

        consumer_list = []
        for i in range(23):
            consumer = Consumer(f'Consumer {i+1}', heat_demand_types, start_times)
            consumer_list.append(consumer)

    else:
        if profile == 'Profile 2':
            heat_demand_types = ['shower', 'shower']
            amount = [5, 13, 5]
            interval = 5 / 60

        elif profile == 'Profile 3':
            heat_demand_types = ['shower', 'shower']
            amount = [1, 2, 4, 9, 4, 2, 1]
            interval = 5 / 60

        elif profile == 'Profile 4':
            heat_demand_types = ['shower', 'shower']
            amount = [1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1]
            interval = 5 / 60

        offsets = interval * np.arange(-len(amount) // 2, len(amount) // 2 + 1)

        start_time1 = peaks[0] + offsets
        start_time2 = peaks[1] + offsets

        consumer_list = []
        tot_num = 0
        for idx, num in enumerate(amount):
            for i in range(num):
                consumer = Consumer(
                    f'Consumer {tot_num+i+1}',
                    heat_demand_types,
                    [start_time1[idx], start_time2[idx]],
                )
                consumer_list.append(consumer)
            tot_num += num

    return consumer_list


def compute_total_heat_demand(profile, peaks, dt=60, total_time=24 * 3600):
    num_steps = total_time // dt
    t_hours = np.arange(num_steps) * dt / 3600

    consumer_list = consumer_start_times(profile, peaks)

    Q_total = np.zeros(num_steps)
    for consumer in consumer_list:
        consumer.initialize_consumer(num_steps, dt)
        Q_total += consumer.Q_d

    return t_hours, Q_total


def plot_heat_demand_profiles():
    peaks = [7.5, 21]
    profiles = ['Profile 1', 'Profile 2', 'Profile 3', 'Profile 4']
    dt = 60          # s
    total_time = 24 * 3600  # s

    thesis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    thesis_fig_folder = os.path.join(thesis_dir, 'Thesis report', 'figures_thesis')

    # Compute total heat demand for each profile
    t_hours = None
    Q_totals = []
    for profile in profiles:
        t, Q = compute_total_heat_demand(profile, peaks, dt, total_time)
        if t_hours is None:
            t_hours = t
        Q_totals.append(Q)

    # --- Individual figures per profile ---
    for idx, (profile, Q) in enumerate(zip(profiles, Q_totals)):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

        ax.plot(t_hours, Q / 1e3, lw=1.5, color='C0')

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Total heat demand (kW)')
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fname = f'heat_demand_profile_{idx + 1}.png'
        plt.savefig(
            os.path.join(thesis_fig_folder, fname),
            dpi=150,
            bbox_inches='tight',
        )
        # plt.show()


if __name__ == '__main__':
    plot_heat_demand_profiles()
