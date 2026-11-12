"""CSV readers and column resolution for the many trajectory file flavours.

Every public odometry dataset writes the same six numbers with a different
header.  Rather than hard-code one layout per dataset, this module reads the
header (when there is one) and resolves columns by pattern, with an explicit
override path for anything unusual.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..traj import Trajectory

__all__ = ["Table", "read_table", "resolve_columns", "table_to_trajectory", "TableError"]


class TableError(ValueError):
    pass


@dataclass
class Table:
    path: Path
    header: list[str] | None
    data: np.ndarray  # (N, C) float64

    @property
    def n_rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.data.shape[1])


def _normalize(name: str) -> str:
    s = name.strip().strip("#%").strip()
    s = re.sub(r"\s*\[[^\]]*\]", "", s)  # drop units: "p_x [m]" -> "p_x"
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _sniff_header(lines: list[str]) -> tuple[list[str] | None, int]:
    """Return (header_names, index_of_first_data_line)."""
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        is_comment = s.startswith("#") or s.startswith("%")
        has_alpha = bool(re.search(r"[A-Za-z]", s))
        has_comma = "," in s
        if is_comment and has_alpha:
            body = s.lstrip("#%").strip()
            names = [_normalize(x) for x in body.split(",")] if has_comma else [_normalize(x) for x in body.split()]
            return names, i + 1
        if not is_comment and has_alpha and has_comma:
            # Headerless-but-named files such as "timestamp,p_x,p_y,..."
            names = [_normalize(x) for x in s.split(",")]
            return names, i + 1
        if not is_comment:
            return None, i
    raise TableError("file has no data lines")


def read_table(path: str | Path) -> Table:
    path = Path(path)
    if not path.exists():
        raise TableError(f"missing file: {path}")

    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()

    header, first = _sniff_header(lines)
    sample = lines[first] if first < len(lines) else ""
    has_comma = "," in sample or (header is not None and len(header) > 1)
    delimiter = "," if has_comma else None

    try:
        data = np.loadtxt(path, delimiter=delimiter, comments="#%", ndmin=2, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - fall back to a tolerant parser
        rows: list[list[float]] = []
        for raw in lines[first:]:
            s = raw.strip()
            if not s or s[0] in "#%":
                continue
            parts = re.split(r"[,\s]+", s)
            try:
                rows.append([float(x) for x in parts if x != ""])
            except ValueError:
                continue
        if not rows:
            raise TableError(f"could not parse numeric data in {path}: {exc}") from exc
        width = min(len(r) for r in rows)
        data = np.asarray([r[:width] for r in rows], dtype=np.float64)

    if data.size == 0:
        raise TableError(f"empty table: {path}")
    if header is not None and len(header) != data.shape[1]:
        # Units in the header can contain commas; trust the data width.
        header = None
    return Table(path=path, header=header, data=data)


_TS_RE = re.compile(r"^(%?time(stamp)?|ts|stamp|t|utime|sec|seconds)(_(ns|us|ms|s))?$")
_P_RE = re.compile(r"^p(osition)?(_[a-z0-9]+)*_(x|y|z)$|^p_(x|y|z)$|^(x|y|z)$")
_Q_RE = re.compile(r"^q(uat|uaternion)?(_[a-z0-9]+)*_(w|x|y|z)$|^q(w|x|y|z)$")
_V_RE = re.compile(r"^v(el|elocity)?(_[a-z0-9]+)*_(x|y|z)$|^v(x|y|z)$|^(vel|velocity)_(x|y|z)$")


def _pick(header: list[str], regex: re.Pattern[str], axis: str) -> int | None:
    for i, name in enumerate(header):
        if regex.match(name) and name.endswith(axis):
            return i
    return None


def resolve_columns(header: list[str] | None, n_cols: int, spec: dict | None = None) -> dict:
    """Map logical fields to column indices.

    ``spec`` may override any of: ``ts``, ``px``, ``py``, ``pz``, ``qw``,
    ``qx``, ``qy``, ``qz``, ``vx``, ``vy``, ``vz`` (1-based, as in spreadsheets).
    """
    if spec:
        out = {k: int(v) - 1 for k, v in spec.items() if v is not None}
        if "ts" not in out:
            raise TableError("column spec must include 'ts'")
        return out

    cols: dict[str, int] = {}
    if header:
        for i, name in enumerate(header):
            if _TS_RE.match(name):
                cols["ts"] = i
                break
        for axis in "xyz":
            idx = _pick(header, _P_RE, axis)
            if idx is not None:
                cols[f"p{axis}"] = idx
        for comp in "wxyz":
            idx = _pick(header, _Q_RE, comp)
            if idx is not None:
                cols[f"q{comp}"] = idx
        for axis in "xyz":
            idx = _pick(header, _V_RE, axis)
            if idx is not None:
                cols[f"v{axis}"] = idx
        if "ts" in cols and all(f"p{a}" in cols for a in "xyz"):
            return cols

    # No usable header: fall back to the two layouts that are actually common.
    # ts + p3 (+ q4) (+ v3), EuRoC / TLIO / Blackbird style.
    if n_cols >= 4:
        guess = {"ts": 0, "px": 1, "py": 2, "pz": 3}
        if n_cols >= 8:
            guess.update({"qw": 4, "qx": 5, "qy": 6, "qz": 7})
            if n_cols >= 11:
                guess.update({"vx": 8, "vy": 9, "vz": 10})
        return guess
    raise TableError(f"cannot infer columns for a {n_cols}-column file without a header")


_TIME_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def detect_time_unit(t: np.ndarray) -> str:
    """Guess s / ms / us / ns from magnitude and spacing."""
    t = np.asarray(t, dtype=np.float64)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if t[0] > 1e17:
        return "ns"
    if t[0] > 1e14:
        return "us"
    if t[0] > 1e11:
        return "ms"
    # Relative clocks starting near zero: decide from the spacing.
    if dt > 1e5:
        return "ns"
    if dt > 1e2:
        return "us"
    if dt > 1.0:
        return "ms"
    return "s"


def table_to_trajectory(
    table: Table,
    columns: dict | None = None,
    time_unit: str = "auto",
    quat_order: str = "wxyz",
    velocity: bool = True,
    name: str | None = None,
) -> Trajectory:
    cols = resolve_columns(table.header, table.n_cols, columns)
    data = table.data

    t_raw = data[:, cols["ts"]]
    unit = detect_time_unit(t_raw) if time_unit == "auto" else time_unit
    t = t_raw * _TIME_UNIT[unit]

    p = np.stack([data[:, cols[f"p{a}"]] for a in "xyz"], axis=1)

    q = None
    if all(f"q{c}" in cols for c in "wxyz"):
        q = np.stack([data[:, cols[f"q{c}"]] for c in "wxyz"], axis=1)
        if quat_order == "xyzw":
            q = q[:, [3, 0, 1, 2]]
        elif quat_order != "wxyz":
            raise ValueError("quat_order must be 'wxyz' or 'xyzw'")

    v = None
    if velocity and all(f"v{a}" in cols for a in "xyz"):
        v = np.stack([data[:, cols[f"v{a}"]] for a in "xyz"], axis=1)

    traj = Trajectory(
        t=t,
        p=p,
        q=q,
        v=v,
        v_frame="world",
        name=name or table.path.stem,
        meta={"path": str(table.path), "time_unit": unit, "header": table.header},
    )
    return traj