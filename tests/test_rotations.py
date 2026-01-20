import numpy as np
from scipy.spatial.transform import Rotation

from robot_sim.rotations import (
    matrix_to_rot6d,
    rot6d_to_matrix,
    rot6d_to_rotvec,
    rotvec_to_rot6d,
)


def test_roundtrip_random_rotations():
    mats = Rotation.random(100, random_state=0).as_matrix()
    assert np.allclose(rot6d_to_matrix(matrix_to_rot6d(mats)), mats, atol=1e-9)


def test_rotvec_roundtrip_matches_rotation():
    rv = Rotation.random(50, random_state=1).as_rotvec()
    back = rot6d_to_rotvec(rotvec_to_rot6d(rv))
    assert np.allclose(
        Rotation.from_rotvec(back).as_matrix(), Rotation.from_rotvec(rv).as_matrix()
    )


def test_gram_schmidt_accepts_unnormalised_input():
    r6 = matrix_to_rot6d(Rotation.random(random_state=2).as_matrix()) * 3.0 + 1e-3
    m = rot6d_to_matrix(r6)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(m), 1.0)


def test_6d_is_continuous_where_axis_angle_flips():
    # Gripper pointing down ≈ rotation of π about x: tiny rotations around it flip the
    # rotation vector's sign, but the 6D representation barely changes.
    a = Rotation.from_rotvec([np.pi - 1e-3, 0, 0])
    b = Rotation.from_rotvec([0.002, 0, 0]) * a
    assert np.linalg.norm(a.as_rotvec() - b.as_rotvec()) > 6.0
    assert (
        np.linalg.norm(matrix_to_rot6d(a.as_matrix()) - matrix_to_rot6d(b.as_matrix()))
        < 0.01
    )
