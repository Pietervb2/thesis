"""
run_bo_no_normalization.py — BO queue runner (no normalization)
================================================================
Runs BO_no_normalization.py for all profiles (1–4) in both the split and
non-split supply temperature bounds variants, using a fixed worker pool.

  split=True  : theta_1 ∈ [60, 62.5],  theta_2 ∈ [62.5, 65]
  split=False : theta_1 = theta_2 ∈ [60, 65]  (unified bounds)

Jobs run in the order defined in JOBS. Set WORKERS to control parallelism
(BO runs are CPU-heavy; 1 is safest, 2 if the machine has headroom).

Usage:
    python run_bo_no_normalization.py
"""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

WORKERS = 4          # number of parallel BO runs at any one time
SCRIPT  = "BO_no_normalization.py"

# All 8 jobs: all split=True first, then all split=False
JOBS = [
    {"label": f"Profile {p} split={s}", "profile": p, "split": s}
    for s in [True, False]
    for p in range(1, 5)
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(label: str, line: str):
    with _print_lock:
        print(f"[{_ts()}] [{label}] {line}", flush=True)


def run_job(job: dict) -> dict:
    label = job["label"]
    # Second CLI arg: "1" = split, "0" = non-split
    args = [sys.executable, SCRIPT, str(job["profile"]), "1" if job["split"] else "0"]

    _log(label, "▶  started")
    started = datetime.now()

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
    total = len(JOBS)
    print(f"\n{'═'*60}")
    print(f"  BO no-normalization runner  —  {total} jobs  —  {WORKERS} workers")
    print(f"{'═'*60}\n", flush=True)

    started_at = datetime.now()
    results    = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_job, job): job for job in JOBS}

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                job = futures[future]
                _log(job["label"], f"✗  EXCEPTION: {exc}")
                results.append({"label": job["label"], "returncode": -1})

    total_elapsed = datetime.now() - started_at
    passed = [r for r in results if r.get("returncode") == 0]
    failed = [r for r in results if r.get("returncode") != 0]

    print(f"\n{'═'*60}")
    print(f"  Done in {total_elapsed.seconds}s  —  "
          f"{len(passed)}/{total} succeeded"
          + (f"  —  {len(failed)} failed" if failed else ""))
    if failed:
        print("\n  Failed jobs:")
        for r in failed:
            print(f"    ✗  {r['label']}")
    print(f"{'═'*60}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
