"""
Rotation representations for absolute end-effector actions.

Absolute actions use the continuous 6D representation (Zhou et al., CVPR 2019): the
first two rows of the rotation matrix. Axis-angle is a poor regression target here:
with the gripper pointing down the rotation is close to π, where the rotation vector
flips sign between neighbouring frames. Diffusion Policy converts robomimic absolute
actions to 6D for the same reason.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def matrix_to_rot6d(mat):
    """(..., 3, 3) rotation matrices → (..., 6): first two rows, flattened."""
    mat = np.asarray(mat)
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def rot6d_to_matrix(rot6d):
    """(..., 6) → (..., 3, 3) via Gram-Schmidt; accepts non-orthonormal network outputs."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def rotvec_to_rot6d(rotvec):
    return matrix_to_rot6d(Rotation.from_rotvec(rotvec).as_matrix())


def rot6d_to_rotvec(rot6d):
    return Rotation.from_matrix(rot6d_to_matrix(rot6d)).as_rotvec()
