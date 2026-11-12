"""Dataset layout knowledge: where the ground truth lives, and how to load it."""

from __future__ import annotations

from pathlib import Path

from ..traj import Trajectory
from .tables import TableError, read_table, table_to_trajectory

__all__ = ["LAYOUTS", "load_trajectory", "find_gt", "find_imu", "discover_sequences"]

# Relative paths inside a sequence folder, per dataset family.
LAYOUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "euroc": {
        "gt": ("mav0/state_groundtruth_estimate0/data.csv", "groundtruth.csv", "gt.csv"),
        "imu": ("mav0/imu0/data.csv",),
        "cam": ("mav0/cam0/data.csv",),
    },
    "tlio": {
        "gt": ("gt/data.csv", "groundtruth.csv", "gt.csv"),
        "imu": ("imu0/data.csv",),
    },
    "blackbird": {
        "gt": ("groundtruth.csv", "gt.csv", "state_groundtruth.csv"),
        "imu": ("imu0/data.csv", "imu.csv"),
    },
    "generic": {
        "gt": ("gt.csv", "groundtruth.csv", "ground_truth.csv", "state_groundtruth_estimate0/data.csv"),
        "imu": ("imu0/data.csv", "imu.csv"),
    },
}


def _first_existing(seq_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        p = seq_dir / rel
        if p.exists():
            return p
    return None


def find_gt(seq_dir: str | Path, fmt: str = "generic") -> Path:
    seq_dir = Path(seq_dir)
    family = LAYOUTS.get(fmt, LAYOUTS["generic"])
    found = _first_existing(seq_dir, family["gt"])
    if found is None:
        raise TableError(f"no ground-truth file under {seq_dir} (tried {family['gt']})")
    return found


def find_imu(seq_dir: str | Path, fmt: str = "generic") -> Path | None:
    seq_dir = Path(seq_dir)
    family = LAYOUTS.get(fmt, LAYOUTS["generic"])
    return _first_existing(seq_dir, family["imu"])


def load_trajectory(
    path: str | Path,
    time_unit: str = "auto",
    quat_order: str = "wxyz",
    columns: dict | None = None,
    name: str | None = None,
) -> Trajectory:
    table = read_table(path)
    return table_to_trajectory(
        table,
        columns=columns,
        time_unit=time_unit,
        quat_order=quat_order,
        name=name or Path(path).parent.name or Path(path).stem,
    )


def discover_sequences(root: str | Path, fmt: str = "euroc") -> list[Path]:
    """Return sequence folders that contain a ground-truth file."""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            find_gt(child, fmt)
        except TableError:
            continue
        out.append(child)
    return out