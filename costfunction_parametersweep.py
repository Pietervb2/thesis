import argparse
import datetime
import os
import numpy as np

from BO import CostFunction
from test import optimization_run


def parse_args():
    parser = argparse.ArgumentParser(
        description='Investigate the cost function by sweeping over a single theta parameter.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Sweep theta_3 from 40000 to 350000 in 20 steps for Profile 2
  python investigate_costfunction_theta.py \\
      --theta-index 3 --min 40000 --max 350000 --steps 20 \\
      --profile 2 \\
      --theta 60 65 null 200000 38.625 3

  # Sweep theta_1 from 10 to 100 in 30 steps for Profile 5
  python investigate_costfunction_theta.py \\
      --theta-index 1 --min 10 --max 100 --steps 30 \\
      --profile 5 \\
      --theta null 65 150000 200000 38.625 3
        """
    )

    parser.add_argument(
        '--theta-index', type=int, required=True, choices=[1, 2, 3, 4, 5, 6],
        help='Which theta to sweep (1–6). This theta will be varied; pass "null" for it in --theta.'
    )
    parser.add_argument(
        '--min', type=float, required=True, dest='range_min',
        help='Minimum value for the swept theta.'
    )
    parser.add_argument(
        '--max', type=float, required=True, dest='range_max',
        help='Maximum value for the swept theta.'
    )
    parser.add_argument(
        '--steps', type=int, default=20,
        help='Number of steps across the range (default: 20).'
    )
    parser.add_argument(
        '--profile', type=int, required=True,
        help='Profile index (e.g. 2 → "Profile 2").'
    )
    parser.add_argument(
        '--theta', nargs=6, required=True, metavar=('T1', 'T2', 'T3', 'T4', 'T5', 'T6'),
        help=(
            'Fixed values for all six thetas. Use "null" for the theta being swept.\n'
            'Order: theta_1 theta_2 theta_3 theta_4 theta_5 theta_6'
        )
    )
    parser.add_argument(
        '--dt', type=int, default=1,
        help='Time step dt (default: 1).'
    )
    parser.add_argument(
        '--pump-pressure', type=float, default=60,
        help='Pump pressure (default: 60).'
    )
    parser.add_argument(
        '--no-curve', action='store_false', dest='curve',
        help='Disable the curve flag (default: curve=True).'
    )

    return parser.parse_args()


def parse_theta_values(raw_values, swept_index):
    """Parse the --theta argument, allowing 'null' for the swept position."""
    result = []
    for i, v in enumerate(raw_values):
        theta_num = i + 1
        if v.lower() == 'null':
            if theta_num != swept_index:
                raise ValueError(
                    f"'null' found at theta_{theta_num}, but --theta-index is {swept_index}. "
                    f"Only the swept theta may be 'null'."
                )
            result.append(None)
        else:
            try:
                result.append(float(v))
            except ValueError:
                raise ValueError(f"Invalid value for theta_{theta_num}: '{v}' (expected a number or 'null')")
    return result


def main():
    args = parse_args()

    start = datetime.datetime.now()
    print(f'Start: {start}')
    print(f'Sweeping theta_{args.theta_index} from {args.range_min} to {args.range_max} in {args.steps} steps')

    profile_index = args.profile
    profile = f'Profile {profile_index}'

    # Parse and validate fixed theta values
    fixed_theta = parse_theta_values(args.theta, args.theta_index)

    # Build the sweep range
    swept_values = np.linspace(args.range_min, args.range_max, args.steps)
    costs = []
    swept_index_0 = args.theta_index - 1  # Convert to 0-based

    for swept_val in swept_values:
        # Assemble full theta array, inserting the swept value at the right position
        theta_list = [swept_val if v is None else v for v in fixed_theta]
        theta = np.array(theta_list)

        test_name = (
            f"cost_function_test_Profile_{profile_index}"
            f"_theta{args.theta_index}_{swept_val}"
        )

        net = optimization_run(
            theta,
            profile,
            args.dt,
            args.pump_pressure,
            args.curve,
            run_type='sweep',
            test_name=test_name,
        )

        cost_function = CostFunction(
            profile,
            args.dt,
            args.pump_pressure,
            args.curve,
            run_type='test',
            test_name=test_name,
        )
        cost = cost_function.compute_cost(net, theta)
        costs.append(cost)

    print(f'Costs: {costs}')

    # Mirror the folder path that Simulation uses for run_type='test':
    #   <base_dir>/figures/simulation/<test_name>_dt=<dt>_Tambt=<T_ambt>/
    # The last swept value's simulation folder is used as the reference to
    # locate <base_dir> (two levels up from this script, same as simulation.py).
    # The CSV is saved one level above the per-value subfolders, in a sweep
    # summary folder that groups the entire run.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_folder = os.path.join(
        base_dir,
        "thesis",
        "figures",
        "sweeps",
        f"cost_function_sweep_Profile_{profile_index}_theta{args.theta_index}_Ts=11",
    )
    os.makedirs(sweep_folder, exist_ok=True)

    output_file = os.path.join(
        sweep_folder,
        f'cost_function_results_Profile_{profile_index}_theta{args.theta_index}.csv',
    )
    np.savetxt(
        output_file,
        np.column_stack([list(swept_values), costs]),
        delimiter=',',
        header=f'theta_{args.theta_index},cost',
        comments='',
    )
    print(f'Results saved to {output_file}')
    print(f'Duration: {datetime.datetime.now() - start}')


if __name__ == '__main__':
    main()