import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_sim.envs import FrankaSimEnv, gripper_from_sim, gripper_to_sim
from robot_sim.rotations import matrix_to_rot6d


@pytest.fixture(scope="module")
def env():
    e = FrankaSimEnv(
        "lift", image_size=32, action_space="abs", max_steps=5, render_size=64
    )
    yield e
    e.close()


def test_gripper_convention():
    # shared convention 1 = open, 0 = closed; robosuite: -1 = open, +1 = close
    assert gripper_to_sim(1.0) == -1.0 and gripper_to_sim(0.0) == 1.0
    assert gripper_from_sim(-1.0) == 1.0 and gripper_from_sim(1.0) == 0.0


def test_observation_format(env):
    obs, info = env.reset(seed=0)
    assert obs["observation.images.agentview"].shape == (32, 32, 3)
    assert obs["observation.images.agentview"].dtype == np.uint8
    assert obs["observation.images.wrist"].shape == (32, 32, 3)
    assert (
        obs["observation.state"].shape == (9,)
        and obs["observation.state"].dtype == np.float32
    )
    assert info["is_success"] is False
    assert env.observation_space.contains(obs)


def test_seeded_reset_is_reproducible(env):
    s0 = env.reset(seed=123)[0]["observation.state"]
    cube0 = env.get_sim_state().copy()
    env.reset(seed=123)
    assert np.allclose(env.get_sim_state()[1:], cube0[1:])
    env.reset(seed=124)
    assert not np.allclose(env.get_sim_state()[1:], cube0[1:])
    assert s0.shape == (9,)


def _hold_action(obs):
    pos, quat = obs["observation.state"][:3], obs["observation.state"][3:7]
    r6 = matrix_to_rot6d(Rotation.from_quat(quat).as_matrix())
    return np.concatenate([pos, r6, [1.0]]).astype(np.float32)


def test_abs_step_and_truncation(env):
    obs, _ = env.reset(seed=0)
    start = obs["observation.state"][:3].copy()
    for _ in range(5):
        obs, _, terminated, truncated, info = env.step(_hold_action(obs))
    assert truncated and not terminated
    # holding the current pose keeps the arm (nearly) still
    assert np.linalg.norm(obs["observation.state"][:3] - start) < 0.01


def test_wrong_action_shape_raises(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(7))


def test_render_is_upright_rgb(env):
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8


@pytest.mark.parametrize("task", ["can", "square"])
def test_every_task_builds_and_steps(task):
    e = FrankaSimEnv(
        task, image_size=32, action_space="abs", max_steps=2, render_size=64
    )
    try:
        obs, info = e.reset(seed=0)
        assert obs["observation.state"].shape == (9,) and info["is_success"] is False
        pos, quat = obs["observation.state"][:3], obs["observation.state"][3:7]
        hold = np.concatenate(
            [pos, matrix_to_rot6d(Rotation.from_quat(quat).as_matrix()), [1.0]]
        )
        e.step(hold)
    finally:
        e.close()
