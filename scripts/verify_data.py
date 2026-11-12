"""Data sanity checks. Run this *before* training, not after.

    python scripts/verify_data.py --root data/euroc --format euroc
    python scripts/verify_data.py --root data/euroc/MH_02_easy --format euroc --deep

What it catches (in rough order of how much time it saves):
  * wrong folder layout          -> "no ground-truth file"
  * nanosecond vs second clocks  -> duration of 1e9 seconds
  * comma vs whitespace columns  -> only 1 column parsed
  * quaternion sign/normalisation-> norms far from 1
  * dropped frames               -> a 3 s hole in the middle of a 90 s sequence
  * duplicate timestamps         -> dt == 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssio_eval.io.datasets import LAYOUTS, find_gt, find_imu  # noqa: E402
from ssio_eval.io.tables import TableError, read_table, table_to_trajectory  # noqa: E402


def _severity(ok: bool) -> str:
    return "OK  " if ok else "WARN"


def check_sequence(seq_dir: Path, fmt: str) -> dict:
    row: dict = {"seq": seq_dir.name, "problems": []}

    try:
        gt_path = find_gt(seq_dir, fmt)
    except TableError as exc:
        row["problems"].append(str(exc))
        return row

    try:
        table = read_table(gt_path)
    except TableError as exc:
        row["problems"].append(f"cannot read gt: {exc}")
        return row

    row["gt_file"] = str(gt_path.relative_to(seq_dir)).replace("\\", "/")
    row["rows"] = table.n_rows
    row["cols"] = table.n_cols
    row["header"] = ",".join(table.header) if table.header else "(none)"

    try:
        traj = table_to_trajectory(table, name=seq_dir.name)
    except Exception as exc:  # noqa: BLE001
        row["problems"].append(f"cannot build trajectory: {exc}")
        return row

    row.update(traj.summary())

    dt = np.diff(traj.t)
    row["dt_p95"] = round(float(np.percentile(dt, 95)), 6)
    row["n_gaps"] = int(np.sum(dt > 10.0 * np.median(dt)))
    row["n_dupes"] = int(np.sum(dt == 0.0))

    if not (1.0 <= traj.duration <= 3 * 3600):
        row["problems"].append(f"implausible duration {traj.duration:.1f}s (unit bug?)")
    if row["n_gaps"]:
        row["problems"].append(f"{row['n_gaps']} gap(s) > 10x median dt")
    if traj.q is not None:
        norms = np.linalg.norm(traj.q, axis=1)
        row["quat_norm_dev"] = round(float(np.max(np.abs(norms - 1.0))), 6)
        if row["quat_norm_dev"] > 1e-3:
            row["problems"].append(f"quaternion norm deviates by {row['quat_norm_dev']:.2e}")
    else:
        row["problems"].append("no orientation columns (body-frame AVE unavailable)")

    imu = find_imu(seq_dir, fmt)
    row["imu"] = "yes" if imu else "no"

    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Check dataset layout and clocks")
    p.add_argument("--root", required=True, help="dataset root, or one sequence folder")
    p.add_argument("--format", default="euroc", choices=sorted(LAYOUTS))
    p.add_argument("--deep", action="store_true", help="also print the first rows of each file")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"missing root: {root}")
        return 2

    candidates = [root]
    if any(child.is_dir() for child in root.iterdir()):
        candidates = sorted(child for child in root.iterdir() if child.is_dir())

    rows = [check_sequence(seq, args.format) for seq in candidates]
    usable = [r for r in rows if "gt_file" in r]
    if not usable:
        print(f"No sequence with a ground-truth file under {root} (format={args.format})")
        print(f"Expected relative paths: {[p[0] for p in LAYOUTS[args.format].values()]}")
        return 2

    header = f"{'sequence':<26s}{'n':>8s}{'dur[s]':>9s}{'dt[ms]':>8s}{'path[m]':>10s}{'gaps':>6s}  status"
    print(header)
    print("-" * len(header))
    bad = 0
    for r in rows:
        if "gt_file" not in r:
            print(f"{r['seq']:<26s}{'-':>8s}{'-':>9s}{'-':>8s}{'-':>10s}{'-':>6s}  FAIL  {r['problems'][0]}")
            bad += 1
            continue
        print(
            f"{r['seq']:<26s}{r['n']:>8d}{r['duration_s']:>9.1f}"
            f"{r['dt_median_s'] * 1e3:>8.2f}{r['path_length_m']:>10.1f}{r['n_gaps']:>6d}  "
            f"{'OK' if not r['problems'] else 'WARN'}"
        )
        for prob in r["problems"]:
            print(f"    - {prob}")
            bad += int("duration" in prob or "quaternion" in prob or "cannot" in prob)

    if args.deep and rows:
        r = next((x for x in rows if "gt_file" in x), None)
        if r:
            print(f"\nfirst sequence: {r['seq']}")
            print(f"  gt file  : {r['gt_file']}")
            print(f"  header   : {r['header']}")
            print(f"  time unit: {r.get('time_unit', 'auto')}")

    print(f"\n{len(usable)} sequence(s) usable, {len(rows) - len(usable)} unusable.")
    if bad:
        print("Fix the WARN/FAIL lines before training; a unit bug silently ruins every number.")
    return 0 if len(usable) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())