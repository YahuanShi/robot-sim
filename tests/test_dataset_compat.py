"""
Dataset ↔ simulator compatibility (needs the robomimic files; skipped if absent).

Run `python -m robot_sim.datasets` or scripts/replay_demos.py once to download.
"""

import os

import numpy as np
import pytest

from robot_sim.convert import replay_and_record
from robot_sim.datasets import dataset_path, demo_keys, env_args, iter_demos
from robot_sim.envs import FrankaSimEnv
from robot_sim.tasks import OSC_POSE_CONFIG, get_task

LIFT = dataset_path("lift")
pytestmark = pytest.mark.skipif(
    not os.path.exists(LIFT), reason="lift dataset not downloaded"
)


def test_controller_config_matches_dataset():
    ea = env_args(LIFT)
    assert ea["env_name"] == get_task("lift").env_name
    rec = ea["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    ours = OSC_POSE_CONFIG["body_parts"]["right"]
    for k in ("type", "output_max", "output_min", "kp", "damping", "input_ref_frame"):
        assert rec[k] == ours[k], k
    assert ea["env_kwargs"]["control_freq"] == 20


def test_demo_replays_with_delta_and_abs_actions():
    demo = next(iter_demos(LIFT, demo_keys(LIFT)[:1]))
    delta_env = FrankaSimEnv("lift", action_space="delta", image_size=32)
    frames, ok = replay_and_record(delta_env, demo)
    assert ok
    assert frames[0]["abs"].shape == (10,) and frames[0]["delta"].shape == (7,)
    # gripper converted to 1 = open: demos start with the gripper open
    assert frames[0]["abs"][-1] == 1.0
    abs_env = FrankaSimEnv("lift", action_space="abs", image_size=32)
    abs_env.reset_to(demo.model_xml, demo.states[0])
    success = False
    for f in frames:
        success |= abs_env.step(f["abs"])[4]["is_success"]
    assert success
    assert np.isfinite(np.stack([f["abs"] for f in frames])).all()
