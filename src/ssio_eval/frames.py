"""SO(3) / quaternion helpers. numpy only (no scipy dependency).

Conventions used everywhere in this package:
  * quaternions are (w, x, y, z), Hamilton product, and map BODY -> WORLD.
  * a rotation matrix ``R`` satisfies ``p_world = R @ p_body``.
  * timestamps are float seconds, strictly increasing.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize the last axis of a (4,) or (N, 4) quaternion array."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(4,) -> (3, 3); (N, 4) -> (N, 3, 3)."""
    q = normalize_quat(q)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)
    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)
    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R[0] if single else R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(3, 3) -> (4,); (N, 3, 3) -> (N, 4). Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    Rs = np.atleast_3d(R).reshape(-1, 3, 3) if single else R.reshape(-1, 3, 3)

    out = np.empty((Rs.shape[0], 4), dtype=np.float64)
    for i, m in enumerate(Rs):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        out[i] = q if q[0] >= 0 else -q  # canonical: w >= 0
    return out[0] if single else out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (*) b, broadcasting over the leading axes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, tau) -> np.ndarray:
    """Element-wise slerp. q0, q1: (N, 4); tau: (N,) in [0, 1]."""
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1, 1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    # sin(theta) is unstable near both 0 and pi; fall back to nlerp when small.
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    s0 = np.where(small, 1.0 - tau, np.sin((1.0 - tau) * theta) / np.where(small, 1.0, sin_theta))
    s1 = np.where(small, tau, np.sin(tau * theta) / np.where(small, 1.0, sin_theta))
    return normalize_quat(s0 * q0 + s1 * q1)


def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def yaw_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Heading angle about +z, i.e. atan2(R[1,0], R[0,0])."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        return np.array(np.arctan2(R[1, 0], R[0, 0]))
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
    return yaw_from_rotmat(quat_to_rotmat(q))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> rotation vector (3,). Small angles handled."""
    R = np.asarray(R, dtype=np.float64)
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-4:
        # Angle near pi: recover the axis from the symmetric part.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        if axis[i] > 1e-8:
            axis = A[:, i] / axis[i]
        axis = axis / np.maximum(np.linalg.norm(axis), _EPS)
        return axis * theta
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )