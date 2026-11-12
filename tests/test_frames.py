"""Tests for the SO(3) helpers."""

from __future__ import annotations

import numpy as np

from ssio_eval import frames


def test_quat_rotmat_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = frames.normalize_quat(rng.normal(size=4))
        R = frames.quat_to_rotmat(q)
        q2 = frames.rotmat_to_quat(R)
        R2 = frames.quat_to_rotmat(q2)
        assert np.allclose(R, R2, atol=1e-10), "quaternion round-trip lost the rotation"
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_rotmat_is_orthonormal_for_batch() -> None:
    rng = np.random.default_rng(1)
    q = frames.normalize_quat(rng.normal(size=(10, 4)))
    R = frames.quat_to_rotmat(q)
    assert R.shape == (10, 3, 3)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.tile(np.eye(3), (10, 1, 1)), atol=1e-10)


def test_slerp_endpoints() -> None:
    q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
    q1 = frames.normalize_quat(np.array([0.3, 0.8, -0.1, 0.2]))
    q1 = np.broadcast_to(q1, (1, 4))
    start = frames.quat_slerp(q0, q1, np.array([0.0]))
    end = frames.quat_slerp(q0, q1, np.array([1.0]))
    assert np.allclose(np.abs(np.sum(start * q0)), 1.0, atol=1e-9)
    assert np.allclose(np.abs(np.sum(end * q1)), 1.0, atol=1e-9)


def test_slerp_is_unit_and_monotone() -> None:
    q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
    angle = np.deg2rad(120.0)
    q1 = np.array([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])
    taus = np.linspace(0.0, 1.0, 13)
    qs = frames.quat_slerp(np.broadcast_to(q0, (13, 4)), np.broadcast_to(q1, (13, 4)), taus)
    assert np.allclose(np.linalg.norm(qs, axis=1), 1.0, atol=1e-10)
    yaws = frames.quat_to_yaw(qs)
    assert np.all(np.diff(yaws) > 0.0), "slerp should sweep the angle monotonically"


def test_rotz_and_yaw() -> None:
    for deg in (-90.0, -30.0, 0.0, 17.0, 91.0):
        R = frames.rotz(np.deg2rad(deg))
        assert np.isclose(np.degrees(frames.yaw_from_rotmat(R)), deg, atol=1e-9)


def test_quat_mul_composition() -> None:
    rng = np.random.default_rng(2)
    qa = frames.normalize_quat(rng.normal(size=4))
    qb = frames.normalize_quat(rng.normal(size=4))
    Rab = frames.quat_to_rotmat(frames.quat_mul(qa, qb))
    assert np.allclose(Rab, frames.quat_to_rotmat(qa) @ frames.quat_to_rotmat(qb), atol=1e-10)


def test_so3_log_small_and_large() -> None:
    R = frames.rotz(1e-9)
    assert np.linalg.norm(frames.so3_log(R)) < 1e-8
    R = frames.rotz(np.pi - 1e-6)
    assert np.isclose(np.linalg.norm(frames.so3_log(R)), np.pi, atol=1e-4)