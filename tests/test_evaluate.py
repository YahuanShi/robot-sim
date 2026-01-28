import os

import numpy as np
from scipy.spatial.transform import Rotation

from robot_sim.evaluate import evaluate, wilson_interval
from robot_sim.rotations import matrix_to_rot6d


class HoldPolicy:
    """Keeps the end effector where it is (never succeeds)."""

    def reset(self, seed=None):
        self.calls = 0
        self.seeds = getattr(self, "seeds", []) + [seed]

    def act(self, obs):
        pos, quat = obs["observation.state"][:3], obs["observation.state"][3:7]
        r6 = matrix_to_rot6d(Rotation.from_quat(quat).as_matrix())
        return np.concatenate([pos, r6, [1.0]])


def test_evaluate_loop_and_videos(tmp_path):
    policy = HoldPolicy()
    result = evaluate(
        policy,
        task="lift",
        n_episodes=2,
        image_size=32,
        max_steps=4,
        video_dir=str(tmp_path / "videos"),
        n_videos=1,
        output_json=str(tmp_path / "result.json"),
    )
    assert result["successes"] == 0 and result["n_episodes"] == 2
    assert [e["seed"] for e in result["episodes"]] == [100_000, 100_001]
    assert policy.seeds == [100_000, 100_001]
    assert all(e["steps"] == 4 for e in result["episodes"])
    assert os.listdir(tmp_path / "videos") == ["seed100000_fail.mp4"]
    assert (tmp_path / "result.json").exists()


def test_wilson_interval():
    lo, hi = wilson_interval(50, 50)
    assert 0.9 < lo and hi == 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
