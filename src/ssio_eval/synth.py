"""Synthetic trajectories with a *known* answer.

Why bother: you cannot debug an evaluator with real data, because you do not
know what the right number is.  Here you do.  Run ``python -m ssio_eval.cli
synth`` and then evaluate the estimate against the ground truth -- the injected
error is what the metric should report back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import frames
from .traj import Trajectory

__all__ = ["figure_eight", "write_euroc_csv", "corrupt", "make_pair"]


def figure_eight(
    duration: float = 90.0,
    rate: float = 200.0,
    radius: float = 6.0,
    period: float = 12.0,
    seed: int = 0,
) -> Trajectory:
    """A smooth, closed figure-eight with varying speed and yaw-only attitude."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * rate)) + 1
    t = np.linspace(0.0, duration, n)
    w = 2.0 * np.pi / period

    # Position and analytic velocity of a Lissajous figure-eight.
    x = radius * np.sin(w * t)
    y = radius * 0.5 * np.sin(2.0 * w * t)
    z = 0.15 * radius * np.sin(0.5 * w * t) + 0.2
    vx = radius * w * np.cos(w * t)
    vy = radius * 0.5 * 2.0 * w * np.cos(2.0 * w * t)
    vz = 0.15 * radius * 0.5 * w * np.cos(0.5 * w * t)

    p = np.stack([x, y, z], axis=1)
    v = np.stack([vx, vy, vz], axis=1)

    yaw = np.arctan2(vy, vx)
    q = np.zeros((n, 4))
    q[:, 0] = np.cos(yaw / 2.0)
    q[:, 3] = np.sin(yaw / 2.0)

    return Trajectory(t=t, p=p, q=q, v=v, v_frame="world", name="synth_gt", meta={"seed": seed, "rate": rate})


def write_euroc_csv(path: str | Path, traj: Trajectory, time_unit: str = "ns") -> Path:
    """Write a trajectory in the EuRoC ground-truth column layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[time_unit]

    header = (
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
        "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1]"
    )
    ts = ((traj.t - traj.t[0]) * scale).astype(np.int64)
    q = traj.q if traj.q is not None else np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(traj), 1))
    v = traj.world_velocity()
    cols = [ts[:, None], traj.p, q, v]
    data = np.concatenate(cols, axis=1)

    fmt = ["%d"] + ["%.9f"] * 10
    np.savetxt(path, data, delimiter=", ", fmt=fmt, header=header, comments="")
    return path


def _smooth_noise(rng, n: int, sigma_samples: float, amp: float) -> np.ndarray:
    """Low-pass noise: white noise would make finite-difference velocity explode."""
    x = rng.normal(0.0, 1.0, size=n)
    half = max(int(3 * sigma_samples), 1)
    grid = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (grid / max(sigma_samples, 1e-6)) ** 2)
    kernel /= kernel.sum()
    x = np.convolve(x, kernel, mode="same")
    return x / max(float(np.std(x)), 1e-9) * amp


def corrupt(
    gt: Trajectory,
    vel_bias_world=(0.05, 0.0, 0.0),
    pos_noise: float = 0.01,
    yaw_offset_deg: float = 0.0,
    scale: float = 1.0,
    seed: int = 1,
) -> Trajectory:
    """Build a plausible estimate: constant velocity offset + smooth position noise.

    ``vel_bias_world`` integrates exactly into ``bias * (t - t0)`` of position
    drift, which is the failure mode the SSIO paper discusses: invisible to a
    velocity-*change* target, visible to ATE.  The estimate carries its own
    world-frame velocity so the metric is not fed a finite-difference artefact.
    """
    rng = np.random.default_rng(seed)
    n = len(gt)
    bias = np.asarray(vel_bias_world, dtype=np.float64).reshape(3)
    dt = gt.t - gt.t[0]

    rate = (n - 1) / max(gt.duration, 1e-9)
    sigma_samples = max(0.15 * rate, 1.0)  # ~0.15 s correlation time
    jitter = np.stack([_smooth_noise(rng, n, sigma_samples, pos_noise) for _ in range(3)], axis=1)

    v_world = scale * gt.world_velocity() + bias[None, :]
    p = scale * gt.p + bias[None, :] * dt[:, None] + jitter

    Rz = frames.rotz(np.deg2rad(yaw_offset_deg))
    p = p @ Rz.T
    v_world = v_world @ Rz.T
    q = None
    if gt.q is not None:
        qz = frames.rotmat_to_quat(Rz)
        q = frames.quat_mul(np.broadcast_to(qz, gt.q.shape), gt.q)
    return Trajectory(t=gt.t.copy(), p=p, q=q, v=v_world, v_frame="world", name="synth_est")


def make_pair(out_dir: str | Path, seed: int = 0, **kwargs) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = figure_eight(seed=seed)
    est = corrupt(gt, seed=seed + 1, **kwargs)
    return write_euroc_csv(out_dir / "gt.csv", gt), write_euroc_csv(out_dir / "est.csv", est)