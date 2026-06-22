"""
run_sweeps.py — parallel theta sweep runner
============================================
Runs investigate_costfunction_theta.py with a pool of 4 workers.
As soon as one sweep finishes, the next one in the queue is picked up
automatically, so exactly 4 sweeps are always running (until the queue runs out).

Usage:
    python run_sweeps.py

Edit the SWEEPS list below to define all the runs you want.
"""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

WORKERS = 2          # number of parallel sweeps at any one time
SCRIPT  = "costfunction_parametersweep.py"

# Define every sweep you want to run.
# Each entry is a dict that maps directly to the CLI arguments of the script.
# Use None for the theta being swept — it becomes "null" automatically.

SWEEPS = [
    {
        "label":       "theta_5 profile_1",
        "theta_index": 5,
        "min":         30,
        "max":         55,
        "steps":       20,
        "profile":     3,
        "theta":       [63, 65, 300000, 200000, None, 3],
    },
    {
        "label":       "theta_1 profile_3",
        "theta_index": 1,
        "min":         60,
        "max":         65,
        "steps":       20,
        "profile":     3,
        "theta":       [None, 65, 150e3, 100e3, 30, 3],
    },
    {
        "label":       "theta_2 profile_3",
        "theta_index": 2,
        "min":         62,
        "max":         65,
        "steps":       20,
        "profile":     3,
        "theta":       [62, None, 150e3, 100e3, 38.625, 3],
    },
    {
        "label":       "theta_1 profile_3",
        "theta_index": 1,
        "min":         60,
        "max":         65,
        "steps":       20,
        "profile":     3,
        "theta":       [None, 65, 150000, 200000, 38.625, 3],
    },
    #   {
    #     "label":       "theta_2 profile_3",
    #     "theta_index": 2,
    #     "min":         60,
    #     "max":         70,
    #     "steps":       20,
    #     "profile":     3,
    #     "theta":       [60, None, 150000, 200000, 38.625, 3],
    # },
    # {
    #     "label":       "theta_3 profile_3",
    #     "theta_index": 3,
    #     "min":         40000,
    #     "max":         350000,
    #     "steps":       20,
    #     "profile":     3,
    #     "theta":       [60, 65, None, 200000, 38.625, 3],
    # },
    # {
    #     "label":       "theta_4 profile_3",
    #     "theta_index": 4,
    #     "min":         0,
    #     "max":         200e3,
    #     "steps":       20,
    #     "profile":     3,
    #     "theta":       [60, 65, 150000, None, 38.625, 3],
    # },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Lock so lines from concurrent workers don't interleave mid-line
_print_lock = threading.Lock()

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(label: str, line: str):
    """Print a prefixed line, thread-safely."""
    with _print_lock:
        print(f"[{_ts()}] [{label}] {line}", flush=True)


def build_args(sweep: dict) -> list[str]:
    """Convert a sweep dict into a CLI argument list."""
    theta_strs = ["null" if v is None else str(v) for v in sweep["theta"]]
    return [
        sys.executable, SCRIPT,
        "--theta-index", str(sweep["theta_index"]),
        "--min",         str(sweep["min"]),
        "--max",         str(sweep["max"]),
        "--steps",       str(sweep["steps"]),
        "--profile",     str(sweep["profile"]),
        "--theta",       *theta_strs,
    ]


def run_sweep(sweep: dict) -> dict:
    """Run a single sweep as a subprocess, streaming its output live."""
    label = sweep["label"]
    args  = build_args(sweep)

    _log(label, "▶  started")

    started = datetime.now()

    # stdout=PIPE + stderr=STDOUT merges both streams so nothing is lost.
    # We read line-by-line so output appears immediately rather than at the end.
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,          # line-buffered
    )

    captured_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        captured_lines.append(line)
        _log(label, line)

    proc.wait()
    elapsed = datetime.now() - started

    status = "✓  done" if proc.returncode == 0 else "✗  FAILED"
    with _print_lock:
        print(f"[{_ts()}] [{label}] {status}  ({elapsed.seconds}s)", flush=True)

    return {
        "label":      label,
        "returncode": proc.returncode,
        "elapsed":    elapsed,
        "output":     "\n".join(captured_lines),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    total = len(SWEEPS)
    print(f"\n{'═'*60}")
    print(f"  Theta sweep runner  —  {total} jobs  —  {WORKERS} workers")
    print(f"{'═'*60}\n", flush=True)

    started_at = datetime.now()
    results    = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_sweep, sweep): sweep for sweep in SWEEPS}

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                sweep = futures[future]
                _log(sweep["label"], f"✗  EXCEPTION: {exc}")
                results.append({"label": sweep["label"], "returncode": -1})

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = datetime.now() - started_at
    passed = [r for r in results if r.get("returncode") == 0]
    failed = [r for r in results if r.get("returncode") != 0]

    print(f"\n{'═'*60}")
    print(f"  Done in {total_elapsed.seconds}s  —  "
          f"{len(passed)}/{total} succeeded"
          + (f"  —  {len(failed)} failed" if failed else ""))
    if failed:
        print("\n  Failed sweeps:")
        for r in failed:
            print(f"    ✗  {r['label']}")
    print(f"{'═'*60}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()