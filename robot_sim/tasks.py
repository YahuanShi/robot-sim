"""
Task registry — robosuite environment settings for each supported task.

The controller config is copied from the robomimic v1.5 datasets' env_args
(OSC_POSE, 20 Hz, world-frame deltas scaled to ±5 cm / ±0.5 rad per step), so an
environment built here is the one the demonstrations were recorded in.
tests/test_dataset_compat.py checks that against the downloaded files.
"""

import copy
from dataclasses import dataclass, field

HF_BASE = (
    "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5"
)

# OSC_POSE controller used by every robomimic v1.5 single-arm dataset
OSC_POSE_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "world",
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    },
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_name: str  # robosuite environment class
    language: str  # task string stored in the LeRobot dataset
    max_steps: int  # rollout budget at evaluation (demos are much shorter)
    env_kwargs: dict = field(default_factory=dict)

    def dataset_url(self, quality="ph"):
        return f"{HF_BASE}/{self.name}/{quality}/demo_v15.hdf5"


TASKS = {
    # Lift a cube off the table. Demos: 36–64 steps.
    "lift": TaskSpec("lift", "Lift", "lift the cube", max_steps=400),
    # Pick a can from the left bin and place it in its slot in the right bin. Demos: 82–151 steps.
    "can": TaskSpec(
        "can",
        "PickPlaceCan",
        "pick up the can and place it in the matching bin",
        max_steps=400,
    ),
}


def get_task(name):
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[name]


def controller_config(action_space="delta"):
    """OSC_POSE config; action_space='abs' makes actions absolute world-frame eef poses."""
    if action_space not in ("delta", "abs"):
        raise ValueError(f"action_space must be 'delta' or 'abs', got {action_space}")
    cfg = copy.deepcopy(OSC_POSE_CONFIG)
    if action_space == "abs":
        cfg["body_parts"]["right"]["input_type"] = "absolute"
    return cfg
