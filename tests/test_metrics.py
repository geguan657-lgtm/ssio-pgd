"""Known-answer tests for the metrics.

Every assertion here has a closed-form expected value, so a failure means the
evaluator is wrong -- not that the model is bad. Keep this file green.
"""

from __future__ import annotations

import numpy as np

from ssio_eval import frames, metrics
from ssio_eval.synth import corrupt, figure_eight
from ssio_eval.traj import Trajectory


def _ramp(n: int = 400, dt: float = 0.01) -> Trajectory:
    """1 m/s along +x, 4 s long."""
    t = np.arange(n) * dt
    p = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return Trajectory(t=t, p=p, q=q)


def test_identity_is_exactly_zero() -> None:
    gt = figure_eight(duration=20.0)
    res = metrics.evaluate(gt, gt, align="none")
    assert res["ate"]["rmse"] < 1e-12
    assert res["ave"]["mean"] < 1e-12


def test_constant_velocity_is_exact() -> None:
    gt = _ramp()
    est = _ramp()
    m = metrics.ave(est, gt, frame="world", mode="none")
    assert m["mean"] < 1e-9, f"constant velocity should have zero AVE, got {m['mean']}"


def test_velocity_bias_is_recovered_in_body_frame() -> None:
    gt = figure_eight(duration=30.0)
    bias = np.array([0.04, -0.03, 0.02])
    est = corrupt(gt, vel_bias_world=bias, pos_noise=0.0)
    m = metrics.ave(est, gt, frame="body", mode="none")
    assert abs(m["mean"] - np.linalg.norm(bias)) < 1e-6


def test_rigid_transform_is_undone_by_se3() -> None:
    gt = figure_eight(duration=30.0)
    R = frames.rotz(np.deg2rad(53.0))
    est = gt.transform(R, np.array([12.0, -7.0, 2.0]))
    m = metrics.ate(est, gt, mode="se3")
    assert m["rmse"] < 1e-9, f"SE(3) alignment failed: {m['rmse']}"


def test_yaw_alignment_recovers_heading_only() -> None:
    gt = figure_eight(duration=30.0)
    R = frames.rotz(np.deg2rad(-41.0))
    est = gt.transform(R, np.array([2.0, 5.0, 0.0]))
    m = metrics.ate(est, gt, mode="yaw")
    assert m["rmse"] < 1e-9
    assert m["yaw_deg"] is not None and abs(abs(m["yaw_deg"]) - 41.0) < 1e-6


def test_scale_requires_sim3() -> None:
    gt = figure_eight(duration=20.0)
    est = gt.transform(np.eye(3), np.zeros(3), 1.07)
    rigid = metrics.ate(est, gt, mode="se3", with_scale=False)
    sim3 = metrics.ate(est, gt, mode="se3", with_scale=True)
    assert sim3["rmse"] < 1e-9, "similarity alignment should absorb a pure scale"
    assert abs(sim3["scale"] - 1 / 1.07) < 1e-6 or abs(sim3["scale"] - 1.07) < 1e-6
    assert rigid["rmse"] > 0.01, "a pure scale must not be absorbed by a rigid fit"


def test_ate_scales_with_drift() -> None:
    gt = figure_eight(duration=20.0)
    small = corrupt(gt, vel_bias_world=(0.01, 0, 0), pos_noise=0.0)
    big = corrupt(gt, vel_bias_world=(0.05, 0, 0), pos_noise=0.0)
    a = metrics.ate(small, gt, mode="se3")["rmse"]
    b = metrics.ate(big, gt, mode="se3")["rmse"]
    assert b > 3.0 * a, f"ATE should grow with drift ({a:.3f} vs {b:.3f})"


def test_rpe_identical_is_zero() -> None:
    gt = figure_eight(duration=20.0)
    m = metrics.rpe(gt, gt, delta=1.0)
    assert m["rmse"] < 1e-9
    assert m["rot_rmse_deg"] < 1e-9
    assert m["n_segments"] > 10


def test_associate_handles_partial_overlap() -> None:
    gt = figure_eight(duration=20.0)
    cut = gt.slice(gt.t[0] + 5.0, gt.t[-1] - 2.0)
    e, g = metrics.associate(cut, gt)
    assert g.t[0] >= gt.t[0] + 5.0 - 1e-9
    assert g.t[-1] <= gt.t[-1] - 2.0 + 1e-9
    assert np.allclose(e.t, g.t)


def test_body_frame_needs_gt_orientation() -> None:
    t = np.linspace(0, 1, 11)
    p = np.stack([t, t, t], axis=1)
    bare_gt = Trajectory(t=t, p=p)
    est = Trajectory(t=t, p=p, q=np.tile(np.array([1.0, 0, 0, 0]), (11, 1)))
    try:
        metrics.ave(est, bare_gt, frame="body")
    except ValueError as exc:
        assert "orientation" in str(exc)
    else:
        raise AssertionError("expected a ValueError when the GT has no orientation")


def test_time_offset_scan_finds_a_shift() -> None:
    gt = figure_eight(duration=20.0)
    shifted = Trajectory(t=gt.t + 0.30, p=gt.p, q=gt.q, name="shifted")
    scan = metrics.best_time_offset(shifted, gt, search=0.6, step=0.05)
    assert abs(scan["best_offset_s"] + 0.30) < 0.11, scan


def test_resample_preserves_linear_motion() -> None:
    gt = _ramp(n=101, dt=0.01)
    coarse = gt.sample(np.arange(0.0, 1.0, 0.1))
    v = coarse.world_velocity()
    assert np.allclose(v[:, 0], 1.0, atol=1e-6)