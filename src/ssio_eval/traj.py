"""Trajectory container: time + position + (optional) orientation / velocity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import frames

__all__ = ["Trajectory"]


@dataclass
class Trajectory:
    """A sampled 6-DoF trajectory.

    Attributes
    ----------
    t : (N,) seconds, will be sorted and de-duplicated.
    p : (N, 3) position in the world frame.
    q : (N, 4) orientation (w, x, y, z) mapping body -> world. Optional.
    v : (N, 3) velocity. Optional; when absent it is derived from ``p``.
    v_frame : "world" or "body" -- how to interpret ``v``.
    """

    t: np.ndarray
    p: np.ndarray
    q: np.ndarray | None = None
    v: np.ndarray | None = None
    v_frame: str = "world"
    name: str = "traj"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64).ravel()
        self.p = np.asarray(self.p, dtype=np.float64).reshape(-1, 3)
        n = self.t.size
        if self.p.shape[0] != n:
            raise ValueError(f"{self.name}: t has {n} samples but p has {self.p.shape[0]}")
        if n < 2:
            raise ValueError(f"{self.name}: need at least 2 samples, got {n}")

        if self.q is not None:
            self.q = frames.normalize_quat(np.asarray(self.q, dtype=np.float64).reshape(-1, 4))
            if self.q.shape[0] != n:
                raise ValueError(f"{self.name}: q has {self.q.shape[0]} samples, expected {n}")
        if self.v is not None:
            self.v = np.asarray(self.v, dtype=np.float64).reshape(-1, 3)
            if self.v.shape[0] != n:
                raise ValueError(f"{self.name}: v has {self.v.shape[0]} samples, expected {n}")
        if self.v_frame not in ("world", "body"):
            raise ValueError("v_frame must be 'world' or 'body'")

        self._sort_and_dedup()

    # ------------------------------------------------------------------ utils
    def _sort_and_dedup(self) -> None:
        order = np.argsort(self.t, kind="stable")
        if not np.array_equal(order, np.arange(self.t.size)):
            self.t, self.p = self.t[order], self.p[order]
            self.q = None if self.q is None else self.q[order]
            self.v = None if self.v is None else self.v[order]
        keep = np.concatenate(([True], np.diff(self.t) > 0.0))
        if not keep.all():
            self.t, self.p = self.t[keep], self.p[keep]
            self.q = None if self.q is None else self.q[keep]
            self.v = None if self.v is None else self.v[keep]

    def __len__(self) -> int:
        return self.t.size

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def path_length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.p, axis=0), axis=1)))

    @property
    def has_orientation(self) -> bool:
        return self.q is not None

    # ------------------------------------------------------------- resampling
    def sample(self, tq: np.ndarray) -> "Trajectory":
        """Linear (position) / slerp (orientation) interpolation at ``tq``."""
        tq = np.asarray(tq, dtype=np.float64).ravel()
        if tq.size == 0:
            raise ValueError("empty query")
        if tq[0] < self.t[0] - 1e-9 or tq[-1] > self.t[-1] + 1e-9:
            raise ValueError(
                f"{self.name}: query [{tq[0]:.6f}, {tq[-1]:.6f}] outside "
                f"trajectory span [{self.t[0]:.6f}, {self.t[-1]:.6f}]"
            )
        tq = np.clip(tq, self.t[0], self.t[-1])

        idx = np.searchsorted(self.t, tq, side="right") - 1
        idx = np.clip(idx, 0, self.t.size - 2)
        t0, t1 = self.t[idx], self.t[idx + 1]
        denom = np.where(t1 > t0, t1 - t0, 1.0)
        tau = np.clip((tq - t0) / denom, 0.0, 1.0)

        p = self.p[idx] + tau[:, None] * (self.p[idx + 1] - self.p[idx])
        q = None
        if self.q is not None:
            q = frames.quat_slerp(self.q[idx], self.q[idx + 1], tau)
        v = None
        if self.v is not None:
            v = self.v[idx] + tau[:, None] * (self.v[idx + 1] - self.v[idx])
        return Trajectory(t=tq, p=p, q=q, v=v, v_frame=self.v_frame, name=f"{self.name}@resampled")

    def slice(self, t0: float, t1: float) -> "Trajectory":
        m = (self.t >= t0) & (self.t <= t1)
        if m.sum() < 2:
            raise ValueError(f"{self.name}: fewer than 2 samples in [{t0}, {t1}]")
        return Trajectory(
            t=self.t[m],
            p=self.p[m],
            q=None if self.q is None else self.q[m],
            v=None if self.v is None else self.v[m],
            v_frame=self.v_frame,
            name=f"{self.name}[{t0:.3f}:{t1:.3f}]",
        )

    # -------------------------------------------------------------- velocities
    def interp_velocity(self) -> np.ndarray:
        """World-frame velocity by finite differences (central, one-sided at ends)."""
        if self.t.size < 2:
            return np.zeros_like(self.p)
        dt = np.diff(self.t)
        dp = np.diff(self.p, axis=0)
        seg_v = dp / dt[:, None]
        v = np.empty_like(self.p)
        v[0] = seg_v[0]
        v[-1] = seg_v[-1]
        if self.t.size > 2:
            v[1:-1] = 0.5 * (seg_v[:-1] + seg_v[1:])
        return v

    def world_velocity(self) -> np.ndarray:
        """Velocity in the world frame, using ``v`` when available."""
        if self.v is None:
            return self.interp_velocity()
        if self.v_frame == "world":
            return self.v
        if self.q is None:
            raise ValueError(f"{self.name}: body-frame velocity needs orientation")
        R = frames.quat_to_rotmat(self.q)
        return np.einsum("nij,nj->ni", R, self.v)

    # ------------------------------------------------------------ transforms
    def transform(self, R: np.ndarray, t: np.ndarray, s: float = 1.0, name: str | None = None) -> "Trajectory":
        """Apply p' = s*R@p + t (and rotate the orientation accordingly)."""
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        p = s * (self.p @ R.T) + t
        q = None
        if self.q is not None:
            qR = frames.rotmat_to_quat(R)
            q = frames.quat_mul(np.broadcast_to(qR, self.q.shape), self.q)
        v = None
        if self.v is not None:
            v = s * (self.v @ R.T) if self.v_frame == "world" else self.v
        return Trajectory(
            t=self.t.copy(),
            p=p,
            q=q,
            v=v,
            v_frame=self.v_frame,
            name=name or f"{self.name}'",
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        dt = np.diff(self.t)
        return {
            "name": self.name,
            "n": int(self.t.size),
            "duration_s": round(self.duration, 3),
            "path_length_m": round(self.path_length, 3),
            "dt_median_s": round(float(np.median(dt)), 6),
            "dt_max_s": round(float(dt.max()), 6),
            "has_orientation": self.has_orientation,
            "has_velocity": self.v is not None,
        }