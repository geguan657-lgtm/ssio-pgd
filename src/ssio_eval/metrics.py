"""Trajectory metrics: ATE, AVE (average velocity error), RPE.

Every metric is implemented with an explicit, documented convention because
"ATE = 0.34 m" means nothing without saying *how* it was aligned and *which*
frames were compared.  See ``docs/METRICS.md`` before quoting any number.

Nothing here is specific to SSIO; it is a generic odometry evaluator.
"""

from __future__ import annotations

import numpy as np

from . import frames
from .traj import Trajectory

__all__ = [
    "associate",
    "umeyama",
    "align_yaw",
    "estimate_alignment",
    "ate",
    "ave",
    "rpe",
    "evaluate",
    "best_time_offset",
]


# --------------------------------------------------------------------- helpers
def associate(est: Trajectory, gt: Trajectory) -> tuple[Trajectory, Trajectory]:
    """Restrict both trajectories to their overlapping time span.

    The estimate is resampled onto the ground-truth timestamps (the GT is
    usually the denser / more trustworthy clock).
    """
    t0 = max(est.t[0], gt.t[0])
    t1 = min(est.t[-1], gt.t[-1])
    if t1 - t0 < 1e-3:
        raise ValueError(
            f"no temporal overlap: est [{est.t[0]:.3f}, {est.t[-1]:.3f}] "
            f"vs gt [{gt.t[0]:.3f}, {gt.t[-1]:.3f}]"
        )
    gt_win = gt.slice(t0, t1)
    if est.t.size == gt_win.t.size and np.allclose(est.t, gt_win.t, rtol=0.0, atol=1e-9):
        est_win = est.slice(gt_win.t[0], gt_win.t[-1])  # already on the same clock
    else:
        est_win = est.sample(gt_win.t)
    return est_win, gt_win


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Returns (R, t, s) such that ``s * R @ src_i + t ~= dst_i``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_c**2) / n
        s = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t, s


def align_yaw(src: np.ndarray, dst: np.ndarray):
    """Heading + translation alignment (no scale, no roll/pitch).

    Odometry already observes gravity, so roll/pitch are usually trusted and
    only the yaw offset is free. This is the convention TLIO-style evaluations
    tend to use.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    a = src[:, 0] - mu_s[0]
    b = src[:, 1] - mu_s[1]
    c = dst[:, 0] - mu_d[0]
    d = dst[:, 1] - mu_d[1]
    num = np.sum(a * d - b * c)
    den = np.sum(a * c + b * d)
    theta = float(np.arctan2(num, den))
    R = frames.rotz(theta)
    t = mu_d - R @ mu_s
    return R, t, 1.0, theta


def estimate_alignment(src: np.ndarray, dst: np.ndarray, mode: str = "se3", with_scale: bool = False):
    """Dispatch to the requested alignment convention."""
    if mode == "none":
        return np.eye(3), np.zeros(3), 1.0, None
    if mode == "yaw":
        return align_yaw(src, dst)
    if mode == "se3":
        R, t, s = umeyama(src, dst, with_scale=with_scale)
        return R, t, s, None
    raise ValueError(f"unknown alignment mode: {mode!r} (use none|yaw|se3)")


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "rmse": float(np.sqrt(np.mean(x**2))),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


# ------------------------------------------------------------------- metrics
def ate(
    est: Trajectory,
    gt: Trajectory,
    mode: str = "se3",
    with_scale: bool = False,
    do_associate: bool = True,
) -> dict:
    """Absolute trajectory error of positions (metres)."""
    e, g = associate(est, gt) if do_associate else (est, gt)
    if e.t.size != g.t.size:
        e, g = associate(e, g)

    R, t, s, theta = estimate_alignment(e.p, g.p, mode=mode, with_scale=with_scale)
    p_al = s * (e.p @ R.T) + t
    err = np.linalg.norm(p_al - g.p, axis=1)

    out = _stats(err)
    out.update(
        {
            "metric": "ate",
            "unit": "m",
            "align": mode,
            "with_scale": bool(with_scale),
            "scale": float(s),
            "yaw_deg": None if theta is None else float(np.degrees(theta)),
            "n": int(err.size),
            "path_length_m": g.path_length,
            "duration_s": g.duration,
            # normalised error: comparable across datasets of different size
            "rmse_pct_of_path": float(100.0 * out["rmse"] / max(g.path_length, 1e-9)),
        }
    )
    return out


def ave(
    est: Trajectory,
    gt: Trajectory,
    frame: str = "world",
    mode: str = "none",
    with_scale: bool = False,
) -> dict:
    """Average velocity error (m/s).

    ``frame="world"``  compare world-frame velocities (requires a consistent
                       heading, hence ``mode`` is applied first).
    ``frame="body"``   express both velocities in the *ground-truth* body
                       frame, which cancels any heading offset and is the right
                       choice when the network predicts body-frame velocity.
    """
    if frame not in ("world", "body"):
        raise ValueError("frame must be 'world' or 'body'")

    e, g = associate(est, gt)
    if g.q is None and frame == "body":
        raise ValueError("frame='body' needs ground-truth orientation")

    if frame == "body":
        # Frame-free by construction: no alignment is applied, so the number
        # cannot be massaged by a lucky alignment.
        v_est_w = e.world_velocity()
        v_gt_w = g.world_velocity()
        Rg = frames.quat_to_rotmat(g.q)  # (N, 3, 3) body->world
        v_est = np.einsum("nji,nj->ni", Rg, v_est_w)  # R^T v_world
        v_gt = np.einsum("nji,nj->ni", Rg, v_gt_w)
    else:
        R, t, s, _ = estimate_alignment(e.p, g.p, mode=mode, with_scale=with_scale)
        e_al = e.transform(R, t, s, name="est_aligned")
        v_est, v_gt = e_al.world_velocity(), g.world_velocity()

    err = np.linalg.norm(v_est - v_gt, axis=1)
    out = _stats(err)
    out.update(
        {
            "metric": "ave",
            "unit": "m/s",
            "frame": frame,
            "align": mode,
            "n": int(err.size),
            "gt_speed_mean": float(np.mean(np.linalg.norm(v_gt, axis=1))),
            "est_speed_mean": float(np.mean(np.linalg.norm(v_est, axis=1))),
            "duration_s": g.duration,
        }
    )
    return out


def rpe(
    est: Trajectory,
    gt: Trajectory,
    delta: float = 1.0,
    by: str = "distance",
) -> dict:
    """Relative pose error over segments of length ``delta``.

    ``by="distance"`` uses segments of ``delta`` metres, ``by="time"`` uses
    ``delta`` seconds. Translation part only (rotation error is reported too
    when both trajectories carry orientation).
    """
    e, g = associate(est, gt)
    n = g.t.size
    pairs: list[tuple[int, int]] = []

    if by == "time":
        j = 0
        for i in range(n):
            j = max(j, i + 1)
            while j < n - 1 and g.t[j] - g.t[i] < delta:
                j += 1
            if j < n and g.t[j] - g.t[i] >= delta * 0.5:
                pairs.append((i, j))
    elif by == "distance":
        seg = np.linalg.norm(np.diff(g.p, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        j = 1
        for i in range(n - 1):
            if j <= i:
                j = i + 1
            while j < n - 1 and cum[j] - cum[i] < delta:
                j += 1
            if j < n and cum[j] - cum[i] >= 0.5 * delta:
                pairs.append((i, j))
    else:
        raise ValueError("by must be 'time' or 'distance'")

    if not pairs:
        raise ValueError("no RPE segment could be formed; check delta")

    t_err, r_err = [], []
    for i, j in pairs:
        # Relative displacement expressed in each trajectory's own start body
        # frame: t_A = R_i^T (p_j - p_i). Comparing these two is frame-free.
        if e.q is not None and g.q is not None:
            Ri = frames.quat_to_rotmat(e.q[i])
            Rg = frames.quat_to_rotmat(g.q[i])
            t_a = Ri.T @ (e.p[j] - e.p[i])
            t_b = Rg.T @ (g.p[j] - g.p[i])
            t_err.append(float(np.linalg.norm(t_a - t_b)))

            # E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j) -> rotation part angle.
            dq_est = frames.quat_mul(_qconj(e.q[i]), e.q[j])
            dq_gt = frames.quat_mul(_qconj(g.q[i]), g.q[j])
            dq_err = frames.quat_mul(_qconj(dq_gt), dq_est)
            R_err = frames.quat_to_rotmat(dq_err)
            r_err.append(float(np.degrees(np.linalg.norm(frames.so3_log(R_err)))))
        else:
            t_err.append(float(np.linalg.norm((e.p[j] - e.p[i]) - (g.p[j] - g.p[i]))))

    out = _stats(np.array(t_err))
    out.update(
        {
            "metric": "rpe",
            "unit": "m",
            "delta": float(delta),
            "by": by,
            "n_segments": len(pairs),
            "rot_rmse_deg": float(np.sqrt(np.mean(np.square(r_err)))) if r_err else None,
            "trans_error_per_meter": float(out["rmse"] / max(delta, 1e-9)) if by == "distance" else None,
        }
    )
    return out


def _qconj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


# ------------------------------------------------------------------ one-stop
def evaluate(
    est: Trajectory,
    gt: Trajectory,
    align: str = "se3",
    with_scale: bool = False,
    ave_frame: str = "world",
    ave_align: str | None = None,
    rpe_delta: float = 1.0,
    include_rpe: bool = False,
) -> dict:
    """Run the standard metric bundle and return a flat dict.

    ``align`` governs ATE only.  ``ave_align`` governs world-frame AVE and
    defaults to ``"yaw"``: fitting a full SE(3) on a drifting trajectory can
    rotate the velocities and inflate AVE, so heading-only is the safer default.
    Body-frame AVE never applies an alignment at all.
    """
    if ave_align is None:
        ave_align = "yaw" if ave_frame == "world" else "none"

    out = {
        "ate": ate(est, gt, mode=align, with_scale=with_scale),
        "ave": ave(est, gt, frame=ave_frame, mode=ave_align, with_scale=with_scale),
    }
    # The complementary view is cheap and often reveals frame mistakes, but it
    # needs ground-truth attitude: skip it rather than fail the whole evaluation.
    alt_frame = "body" if ave_frame == "world" else "world"
    if alt_frame == "body" and not gt.has_orientation:
        out["ave_alt"] = None
    else:
        out["ave_alt"] = ave(est, gt, frame=alt_frame, mode="none" if alt_frame == "body" else ave_align)
    if include_rpe:
        out["rpe"] = rpe(est, gt, delta=rpe_delta)
    return out


def best_time_offset(
    est: Trajectory,
    gt: Trajectory,
    search: float = 0.5,
    step: float = 0.02,
    mode: str = "se3",
) -> dict:
    """Diagnostic only: scan a constant time shift and report the best one.

    Never report a shifted number as your headline result unless the official
    evaluation does the same -- it hides a real bug otherwise.
    """
    offsets = np.arange(-search, search + 1e-9, step)
    rows = []
    for d in offsets:
        shifted = Trajectory(
            t=est.t + d,
            p=est.p,
            q=est.q,
            v=est.v,
            v_frame=est.v_frame,
            name=f"{est.name}+{d:+.2f}s",
        )
        try:
            m = ate(shifted, gt, mode=mode)
        except ValueError:
            continue
        rows.append((float(d), m["rmse"]))
    if not rows:
        raise ValueError("time-offset scan found no valid overlap")
    best = min(rows, key=lambda r: r[1])
    return {
        "best_offset_s": best[0],
        "best_ate_rmse": best[1],
        "zero_offset_ate_rmse": dict(rows).get(0.0),
        "n": len(rows),
    }