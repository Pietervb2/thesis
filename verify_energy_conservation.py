"""
Verify nodal energy conservation of a completed district-heating simulation.

This is a pure post-processing check: it only reads the CSV files written by
Simulation.save_data() (see baseclasses/simulation.py) under

    <folder>/simulation_data/
    <folder>/hex_consumer_data/

It never touches the live Network/Node objects, so it works on any run that
is still sitting under figures/simulation/...

Four independent checks are performed.

1. Heat-exchanger nodes ('Hex' in the node id)
   ------------------------------------------
   HeatExchanger.NTU_method computes a single Q that is used for both sides,
   so this mostly checks that Hex X.csv was written correctly, and catches
   any future change that breaks that symmetry:

       Q_hot  = mflow_prim * cp * (Th_in - Th_out)
       Q_cold = mflow_sec  * cp * (Tc_out - Tc_in)
       residual = Q_hot - Q_cold

2. All other nodes - energy
   -------------------------
   Node.set_T() sets the node temperature to the massflow-weighted average of
   the *incoming* pipe temperatures only. That average conserves energy only
   if the mass leaving the node equals the mass entering it. This recomputes
   the balance independently, weighting the *outgoing* pipes' massflows by
   the stored node temperature instead:

       residual = cp * [ sum_in(mflow_in * T_pipe_in) - sum_out(mflow_out * T_node) ]

   A non-zero residual here means mass is not conserved at that node (a
   hydraulic-solve bug), since the mixing arithmetic itself is exact.

3. Kirchhoff's Current Law (mass conservation, all non-Hex nodes)
   ----------------------------------------------------------------
   The same imbalance as check 2, but in kg/s instead of W, with no
   temperature involved at all:

       residual = sum_in(mflow_in) - sum_out(mflow_out)

   This isolates whether Network.set_mflow_network's root-finder actually
   conserves mass at every node, independent of temperature - check 2's
   residual is exactly this value scaled by cp * T_node, so the two checks
   should agree on *where* the problem is, if there is one.

4. Kirchhoff's Voltage Law (head-loss balance around every loop)
   -------------------------------------------------------------
   Network.res() drives, for every independent loop L,

       sum_{pipe in L} (friction[pipe] + elevation[pipe]) - pump_delivered = 0

   This reconstructs that same sum per loop from the saved pressure-drop
   CSVs (see verify_kirchhoff_voltage for the reconstruction and its
   caveats) to check that the hydraulic solve actually balances head around
   every loop, not just mass at every node.

Note: Node_temp.csv/Pipe_temp.csv/Pipe_mflow.csv are rounded (3-5 decimals)
before being written, so residuals will not be exactly zero even for a
perfectly conservative simulation - expect O(1e-2 - 1e0) W of rounding noise.
"""

import argparse
import os
import re

import networkx as nx
import numpy as np
import pandas as pd

C_P_WATER = 4186.0  # J/(kg K), matches baseclasses/pipe.py and baseclasses/consumer.py


def _token_to_node_id(node_columns):
    """
    Map the bare token used in Node_dT.csv column names (e.g. '1.2' or '5')
    back to the full node id ('Node 1.2' or 'Hex 5'), using the node ids that
    are actually present in Node_temp.csv.
    """
    token_map = {}
    for node_id in node_columns:
        token = node_id.split()[1]
        token_map[token] = node_id
    return token_map


def build_pipe_topology(node_dT_columns, pipe_columns, node_columns):
    """
    Node_dT.csv and Pipe_temp.csv/Pipe_mflow.csv are written from the same
    'for pipe_id, pipe_info in network.pipes.items()' loop in
    Simulation.save_data, so the k-th dT column (after 'time') lines up with
    the k-th pipe column (after 'time') and encodes that pipe's (from, to).

    Returns: dict pipe_id -> (from_node_id, to_node_id)
    """
    if len(pipe_columns) != len(node_dT_columns):
        raise ValueError(
            f"Pipe_temp.csv has {len(pipe_columns)} pipe columns but Node_dT.csv "
            f"has {len(node_dT_columns)} dT columns - cannot align topology."
        )

    token_map = _token_to_node_id(node_columns)
    dT_pattern = re.compile(r"^dT (.+)_(.+)$")

    topology = {}
    for pipe_id, col in zip(pipe_columns, node_dT_columns):
        match = dT_pattern.match(col)
        if not match:
            raise ValueError(f"Could not parse '{col}' as a 'dT <from>_<to>' column")
        from_token, to_token = match.groups()
        if from_token not in token_map or to_token not in token_map:
            raise ValueError(
                f"Column '{col}' references node token(s) not found in Node_temp.csv "
                f"({from_token!r}, {to_token!r})"
            )
        topology[pipe_id] = (token_map[from_token], token_map[to_token])

    return topology


def load_simulation_data(folder):
    """Load the CSVs needed for the energy-conservation checks from `folder`."""

    sim_data_folder = os.path.join(folder, "simulation_data")

    node_temp = pd.read_csv(os.path.join(sim_data_folder, "Node_temp.csv"))
    node_dT = pd.read_csv(os.path.join(sim_data_folder, "Node_dT.csv"))
    pipe_temp = pd.read_csv(os.path.join(sim_data_folder, "Pipe_temp.csv"))
    pipe_mflow = pd.read_csv(os.path.join(sim_data_folder, "Pipe_mflow.csv"))

    node_columns = [c for c in node_temp.columns if c not in ("time", "T_in", "T_ambient")]
    pipe_columns = [c for c in pipe_temp.columns if c != "time"]
    dT_columns = [c for c in node_dT.columns if c != "time"]

    topology = build_pipe_topology(dT_columns, pipe_columns, node_columns)

    return {
        "node_temp": node_temp,
        "pipe_temp": pipe_temp,
        "pipe_mflow": pipe_mflow,
        "topology": topology,
        "node_columns": node_columns,
    }


def verify_normal_nodes(data):
    """
    Recompute the mixing residual (see module docstring, check 2) for every
    node that is not a heat exchanger.

    Returns a DataFrame indexed by time, one column of residuals [W] per node.
    """
    node_temp = data["node_temp"]
    pipe_temp = data["pipe_temp"]
    pipe_mflow = data["pipe_mflow"]
    topology = data["topology"]

    residuals = {}

    for node_id in data["node_columns"]:
        if "Hex" in node_id:
            continue

        pipes_in = [p for p, (frm, to) in topology.items() if to == node_id]
        pipes_out = [p for p, (frm, to) in topology.items() if frm == node_id]

        if not pipes_in and not pipes_out:
            continue

        T_node = node_temp[node_id].to_numpy()

        energy_in = np.zeros_like(T_node)
        for pipe_id in pipes_in:
            energy_in += pipe_mflow[pipe_id].to_numpy() * pipe_temp[pipe_id].to_numpy()

        energy_out = np.zeros_like(T_node)
        for pipe_id in pipes_out:
            energy_out += pipe_mflow[pipe_id].to_numpy() * T_node

        residuals[node_id] = C_P_WATER * (energy_in - energy_out)

    df = pd.DataFrame(residuals)
    df.insert(0, "time", node_temp["time"].to_numpy())
    return df


def verify_mass_conservation(data):
    """
    Kirchhoff's Current Law (see module docstring, check 3): pure mass-flow
    balance at every non-Hex node, with no temperature involved.

        residual = sum_in(mflow_in) - sum_out(mflow_out)   [kg/s]

    Returns a DataFrame indexed by time, one column of residuals [kg/s] per
    non-Hex node.
    """
    pipe_mflow = data["pipe_mflow"]
    topology = data["topology"]

    n = len(pipe_mflow)
    residuals = {}

    for node_id in data["node_columns"]:
        if "Hex" in node_id:
            continue

        pipes_in = [p for p, (frm, to) in topology.items() if to == node_id]
        pipes_out = [p for p, (frm, to) in topology.items() if frm == node_id]

        if not pipes_in and not pipes_out:
            continue

        mflow_in = np.zeros(n)
        for pipe_id in pipes_in:
            mflow_in += pipe_mflow[pipe_id].to_numpy()

        mflow_out = np.zeros(n)
        for pipe_id in pipes_out:
            mflow_out += pipe_mflow[pipe_id].to_numpy()

        residuals[node_id] = mflow_in - mflow_out

    df = pd.DataFrame(residuals)
    df.insert(0, "time", pipe_mflow["time"].to_numpy())
    return df


def find_network_loops(topology):
    """
    Rebuild the same directed pipe graph as
    Network.build_incidence_matrix_and_graph (baseclasses/network.py) purely
    from pipe topology, and enumerate its independent loops the way
    Network.build_loop_matrix does, so Kirchhoff's Voltage Law can be checked
    without touching the live Network object.

    Every pipe direction is fixed to the network's assumed (always-positive)
    flow direction, so - exactly like build_loop_matrix - every edge in a
    loop contributes with the same sign regardless of which way the loop is
    traversed; there's no separate sign column to track.

    Returns: list of lists of pipe_id, one list per independent loop.
    """
    G = nx.DiGraph()
    for pipe_id, (frm, to) in topology.items():
        G.add_edge(frm, to, id=pipe_id)

    cycles = sorted(nx.simple_cycles(G), key=len)

    loops = []
    for cycle in cycles:
        pipe_ids = []
        for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
            if G.has_edge(u, v):
                pipe_ids.append(G[u][v]["id"])
            elif G.has_edge(v, u):
                pipe_ids.append(G[v][u]["id"])
        loops.append(pipe_ids)
    return loops


def load_hydraulic_data(folder):
    """Load the extra CSVs needed for the Kirchhoff's Voltage Law check."""
    sim_data_folder = os.path.join(folder, "simulation_data")
    hex_folder = os.path.join(folder, "hex_consumer_data")

    hex_dp_file = os.path.join(hex_folder, "Hex_dp.csv")
    hex_valve_file = os.path.join(hex_folder, "Hex_valve_data.csv")
    bypass_file = os.path.join(hex_folder, "bypass.csv")

    return {
        "pipe_dp_friction": pd.read_csv(os.path.join(sim_data_folder, "Pipe_dp_friction.csv")),
        "pipe_dp_head": pd.read_csv(os.path.join(sim_data_folder, "Pipe_dp_head.csv")),
        "hex_dp": pd.read_csv(hex_dp_file) if os.path.exists(hex_dp_file) else None,
        "hex_valve": pd.read_csv(hex_valve_file) if os.path.exists(hex_valve_file) else None,
        "bypass": pd.read_csv(bypass_file) if os.path.exists(bypass_file) else None,
    }


def _build_valve_dp_lookup(hyd):
    """
    Map pipe_id -> array of dP [Pa] contributed by a control valve, and (for
    HEX inlet pipes) the lumped HEX pressure drop - i.e. the
    inv_Kv_array**2 * mflow**2 (+ Kp_array * mflow**2) terms Network folds
    into friction_vector (build_help_vectors_NR / update_valves). Both are
    already saved pre-multiplied by mflow**2 at the correct (current-step,
    not lagged) mass flow, so they're used as-is.
    """
    valve_dp = {}

    hex_dp, hex_valve = hyd["hex_dp"], hyd["hex_valve"]
    if hex_dp is not None:
        for col in hex_dp.columns:  # 'Valve <n>'
            n = col.split()[-1]
            pipe_id = f"Pipe {n}.3"
            kp_term = hex_dp[col].to_numpy(dtype=float)

            dP_col = f"dP {col}"
            if hex_valve is not None and dP_col in hex_valve.columns:
                kv_term = hex_valve[dP_col].fillna(0.0).to_numpy(dtype=float)
            else:
                kv_term = 0.0

            valve_dp[pipe_id] = kp_term + kv_term

    bypass = hyd["bypass"]
    if bypass is not None and "dp" in bypass.columns:
        # The bypass valve's own resistance sits on 'Bypass 2' (Network.add_bypass
        # constructs Valve(..., pipe_id=bypass_id) with bypass_id='Bypass 2').
        valve_dp["Bypass 2"] = bypass["dp"].fillna(0.0).to_numpy(dtype=float)

    return valve_dp


HEX_PIPE_FRICTION_COEFF = 0.1  # Pipe.pressure_friction() hardcodes this for hex_pipe=True pipes


def _pipe_dp_used(pipe_id, pipe_mflow, hyd, hex_pipe_ids, valve_dp):
    """
    Reconstruct the pressure-drop term Network.res() actually used for this
    pipe during the hydraulic solve at every timestep N:

        dp_used[N] = friction_coeff(mflow[N-1]) * mflow[N]**2   (lagged coefficient,
                                                                   see Network.update_friction_vector)
                    + elevation                                  (constant)
                    + valve/HEX dp[N]                             (only where applicable)

    Pipe.pressure_friction() hardcodes a constant 0.1 friction coefficient
    for any hex_pipe=True pipe (the '.3'/'.4' pipes either side of a HEX)
    regardless of flow, so those skip the lagged-coefficient reconstruction.

    For regular pipes, only the *product* coeff*mflow_prev**2 was saved
    (Pipe_dp_friction.csv) - the coefficient itself depends on pipe
    roughness, which isn't saved anywhere - so it's recovered as
    saved_value * (mflow[N] / mflow[N-1])**2. That's undefined the instant a
    pipe's flow goes from ~0 to nonzero (nothing to scale), so that single
    step is marked invalid rather than silently guessed at.

    Returns: (dp_used [Pa], valid [bool]) arrays, same length as pipe_mflow.
    """
    mflow = pipe_mflow[pipe_id].to_numpy(dtype=float)
    mflow_prev = np.concatenate([[np.nan], mflow[:-1]])

    if pipe_id in hex_pipe_ids:
        friction = HEX_PIPE_FRICTION_COEFF * mflow**2
        valid = np.ones_like(mflow, dtype=bool)
    else:
        dp_fric_saved = hyd["pipe_dp_friction"][pipe_id].to_numpy(dtype=float)
        has_prev = mflow_prev > 1e-9
        safe_prev = np.where(has_prev, mflow_prev, 1.0)
        friction = np.where(has_prev, dp_fric_saved * (mflow / safe_prev) ** 2, 0.0)
        valid = has_prev

    elevation = hyd["pipe_dp_head"][pipe_id].iloc[0]

    dp = friction + elevation + valve_dp.get(pipe_id, 0.0)

    return dp, valid


def verify_kirchhoff_voltage(folder, data):
    """
    Kirchhoff's Voltage Law (see module docstring, check 4): sum of pressure
    drop (friction + elevation + valve/HEX terms) around every closed loop
    must equal the pressure the pump delivers into that loop - this is
    Network.res()'s `head_loss - pump_term`, which set_mflow_network's
    root-finder drives to zero.

    Loops are the same ones Network.build_loop_matrix would find (one per
    consumer branch: out from Pump 1 along the supply riser, through that
    consumer's HEX, back along the return riser), rebuilt here purely from
    pipe topology via find_network_loops.

    Caveat: the pump pipe's own friction/elevation is computed internally by
    the simulation but never written to a CSV (only its delivered head,
    pump.a*mflow**2 + pump.b*mflow + pump.c, is saved), so it's excluded
    here. Every loop shares the same pump pipe, so this shows up as a
    similar, roughly-constant offset across all loops rather than spurious
    per-loop or per-timestep variation - the diagnostically useful signal
    (spikes / trends against that baseline) is unaffected.

    Returns a DataFrame indexed by time, one column of residuals [Pa] per
    loop. Values are NaN wherever a pipe in that loop wasn't carrying flow
    (valve shut) or its friction coefficient couldn't be recovered (see
    _pipe_dp_used).
    """
    pipe_mflow = data["pipe_mflow"]
    topology = data["topology"]

    hex_node_ids = {n for n in data["node_columns"] if "Hex" in n}
    hex_pipe_ids = {pid for pid, (frm, to) in topology.items() if frm in hex_node_ids or to in hex_node_ids}

    hyd = load_hydraulic_data(folder)
    valve_dp = _build_valve_dp_lookup(hyd)

    loops = find_network_loops(topology)

    n = len(pipe_mflow)
    residuals = {}

    for i, loop_pipes in enumerate(loops):
        total = np.zeros(n)
        valid = np.ones(n, dtype=bool)

        for pipe_id in loop_pipes:
            if "Pump" in pipe_id:
                total -= hyd["pipe_dp_friction"][f"{pipe_id} dp deliv"].to_numpy(dtype=float)
                continue

            dp, pipe_valid = _pipe_dp_used(pipe_id, pipe_mflow, hyd, hex_pipe_ids, valve_dp)
            total += dp
            valid &= pipe_valid
            valid &= pipe_mflow[pipe_id].to_numpy(dtype=float) > 1e-6

        label = " > ".join(pid.replace("Pipe ", "") for pid in loop_pipes)
        residuals[f"Loop {i + 1} [{label}]"] = np.where(valid, total, np.nan)

    df = pd.DataFrame(residuals)
    df.insert(0, "time", pipe_mflow["time"].to_numpy())
    return df


def verify_heat_exchangers(folder, data):
    """
    Recompute the hot-side vs cold-side heat-transfer residual (see module
    docstring, check 1) for every Hex node.

    Returns a DataFrame indexed by time, one column of residuals [W] per Hex.
    """
    hex_folder = os.path.join(folder, "hex_consumer_data")
    time = data["node_temp"]["time"].to_numpy()

    residuals = {}

    for node_id in data["node_columns"]:
        if "Hex" not in node_id:
            continue

        hex_file = os.path.join(hex_folder, f"{node_id}.csv")
        if not os.path.exists(hex_file):
            continue

        df_hex = pd.read_csv(hex_file)

        Q_hot = df_hex["mflow_prim"] * C_P_WATER * (df_hex["Th_in"] - df_hex["Th_out"])
        Q_cold = df_hex["mflow_sec"] * C_P_WATER * (df_hex["Tc_out"] - df_hex["Tc_in"])

        residuals[node_id] = (Q_hot - Q_cold).to_numpy()

    df = pd.DataFrame(residuals)
    df.insert(0, "time", time[: len(df)])
    return df


def summarize(df, label, unit="W"):
    """Print max/mean absolute residual per column (excluding 'time')."""
    print(f"\n{label}")
    print("-" * len(label))
    cols = [c for c in df.columns if c != "time"]
    if not cols:
        print("  (no nodes found)")
        return

    abs_df = df[cols].abs()
    max_col, mean_col = f"max_abs_residual_{unit}", f"mean_abs_residual_{unit}"

    rows = {}
    for col in cols:
        series = abs_df[col]
        if series.notna().sum() == 0:
            rows[col] = {max_col: np.nan, mean_col: np.nan, "time_at_max": np.nan}
            continue
        idx = series.idxmax()
        rows[col] = {
            max_col: series.max(),
            mean_col: series.mean(),
            "time_at_max": df["time"].loc[idx],
        }

    summary = pd.DataFrame.from_dict(rows, orient="index").sort_values(max_col, ascending=False, na_position="last")
    print(summary.to_string())


def verify_energy_conservation(folder, tolerance_W=1.0, tolerance_kgs=1e-3, tolerance_Pa=500.0, save_csv=True):
    """
    Run all four conservation checks for a simulation output folder (see
    module docstring).

    Args:
        folder: path to a simulation run, e.g.
            'figures/simulation/Kvleak/network=Network Rutger_dt=60_..._Tambt=20'
        tolerance_W: energy residuals with max abs value above this are flagged
        tolerance_kgs: mass-flow (KCL) residuals with max abs value above this are flagged
        tolerance_Pa: loop head-loss (KVL) residuals with max abs value above this are flagged
        save_csv: if True, write the residual tables next to the source data

    Returns: (node_energy_residuals, hex_residuals, node_mass_residuals, loop_residuals)
    """
    data = load_simulation_data(folder)

    node_residuals = verify_normal_nodes(data)
    hex_residuals = verify_heat_exchangers(folder, data)
    mass_residuals = verify_mass_conservation(data)
    loop_residuals = verify_kirchhoff_voltage(folder, data)

    summarize(node_residuals, "Normal node mixing residuals (energy)", unit="W")
    summarize(hex_residuals, "Heat exchanger hot/cold residuals", unit="W")
    summarize(mass_residuals, "Kirchhoff's Current Law - node mass-flow residuals", unit="kgs")
    summarize(loop_residuals, "Kirchhoff's Voltage Law - loop head-loss residuals", unit="Pa")

    flagged = []
    for df, kind, tolerance in (
        (node_residuals, "node energy", tolerance_W),
        (hex_residuals, "hex", tolerance_W),
        (mass_residuals, "node mass (KCL)", tolerance_kgs),
        (loop_residuals, "loop (KVL)", tolerance_Pa),
    ):
        for col in df.columns:
            if col == "time":
                continue
            max_abs = df[col].abs().max()
            if pd.notna(max_abs) and max_abs > tolerance:
                flagged.append((kind, col, max_abs, tolerance))

    if flagged:
        print(f"\n{len(flagged)} check(s) exceed their tolerance:")
        for kind, col, max_abs, tolerance in sorted(flagged, key=lambda r: -r[2] / r[3]):
            print(f"  [{kind}] {col}: {max_abs:.3f} (tolerance {tolerance})")
    else:
        print("\nAll checks within tolerance - mass, energy and head-loss conservation hold.")

    if save_csv:
        out_folder = os.path.join(folder, "simulation_data")
        node_residuals.to_csv(os.path.join(out_folder, "Node_energy_residual.csv"), index=False)
        hex_residuals.to_csv(os.path.join(out_folder, "Hex_energy_residual.csv"), index=False)
        mass_residuals.to_csv(os.path.join(out_folder, "Node_mass_residual.csv"), index=False)
        loop_residuals.to_csv(os.path.join(out_folder, "Loop_pressure_residual.csv"), index=False)

    return node_residuals, hex_residuals, mass_residuals, loop_residuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Simulation run folder, e.g. figures/simulation/Kvleak/network=...")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Flag energy residuals above this many watts")
    parser.add_argument("--tolerance-kgs", type=float, default=1e-3, help="Flag KCL (mass) residuals above this many kg/s")
    parser.add_argument("--tolerance-pa", type=float, default=500.0, help="Flag KVL (loop) residuals above this many Pa")
    parser.add_argument("--no-save", action="store_true", help="Don't write residual CSVs back to disk")
    args = parser.parse_args()


    thesis_folder = os.path.abspath(__file__)
    # folder = os.path.join(os.path.dirname(thesis_folder), 'figures', 'simulation', args.folder)
    folder = os.path.join(os.path.dirname(thesis_folder), 'figures', 'optimization_set', '2026-07-07', args.folder)
    verify_energy_conservation(
        folder,
        tolerance_W=args.tolerance,
        tolerance_kgs=args.tolerance_kgs,
        tolerance_Pa=args.tolerance_pa,
        save_csv=not args.no_save,
    )


# figures/optimization_set/2026-07-08/Profile 3_dt=1_init_points=10_n_iter=15
