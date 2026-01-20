"""
robomimic demo_v15.hdf5 → LeRobot v3 dataset (rendered images + delta or absolute actions).

For every demonstration:
  1. restore the recorded initial scene (model XML + MuJoCo state)
  2. replay the recorded delta actions open-loop in the installed robosuite version
  3. at each step render both cameras *before* acting, and after acting read the
     OSC controller's goal pose — that goal is the absolute action for this step
  4. keep the demo only if the replay succeeds

Rendering during the replay (instead of restoring each recorded state) keeps images,
states and absolute actions on one consistent trajectory, even where contact physics
drifts slightly from the original recording.

Gripper actions are converted to the shared convention 1 = open, 0 = closed.

Usage:
    python -m robot_sim.convert --task lift --root data/lerobot/lift_abs_84
    python -m robot_sim.convert --task can --action_space delta --image_size 96 \
        --root data/lerobot/can_delta_96
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from robot_sim.datasets import download, iter_demos
from robot_sim.envs import (
    ACTION_DIM,
    CAMERAS,
    STATE_DIM,
    FrankaSimEnv,
    gripper_from_sim,
)
from robot_sim.rotations import matrix_to_rot6d
from robot_sim.tasks import get_task

ACTION_NAMES = {
    "abs": [
        "x",
        "y",
        "z",
        "r6d_0",
        "r6d_1",
        "r6d_2",
        "r6d_3",
        "r6d_4",
        "r6d_5",
        "gripper",
    ],
    "delta": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
}
STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
    "finger_l",
    "finger_r",
]


def features(image_size, action_space):
    feats = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }
        for name in CAMERAS
    }
    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": STATE_NAMES,
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (ACTION_DIM[action_space],),
        "names": ACTION_NAMES[action_space],
    }
    return feats


def replay_and_record(env, demo):
    """
    Replay one demo in a delta-action env; returns (frames, success).

    frames: list of dicts with the observation before each step and both action
    encodings of that step ('delta' 7-dim and 'abs' 10-dim, shared gripper convention).
    """
    obs = env.reset_to(demo.model_xml, demo.states[0])
    frames, success = [], False
    ctrl = env.arm_controller
    for sim_action in demo.actions:
        next_obs, *_ = env.step_sim(sim_action)
        gripper = gripper_from_sim(sim_action[6])
        # Goal the controller tracked for this step = absolute version of the delta action
        abs_action = np.concatenate(
            [ctrl.goal_pos, matrix_to_rot6d(ctrl.goal_ori), [gripper]]
        ).astype(np.float32)
        delta_action = np.concatenate([sim_action[:6], [gripper]]).astype(np.float32)
        frames.append({"obs": obs, "abs": abs_action, "delta": delta_action})
        obs = next_obs
        success |= env.is_success()
    return frames, success


def convert(
    task,
    root,
    repo_id=None,
    quality="ph",
    image_size=84,
    action_space="abs",
    max_demos=None,
    overwrite=False,
    vcodec="h264",
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = get_task(task)
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise SystemExit(f"{root} exists — pass --overwrite to re-create it")
        shutil.rmtree(root)

    path = download(task, quality)
    demos = list(iter_demos(path))[:max_demos]
    env = FrankaSimEnv(task, action_space="delta", image_size=image_size)

    ds = LeRobotDataset.create(
        repo_id=repo_id
        or f"local/robomimic_{task}_{quality}_{action_space}_{image_size}",
        fps=20,
        root=root,
        features=features(image_size, action_space),
        use_videos=True,
        vcodec=vcodec,
    )
    skipped = []
    for demo in tqdm(demos, desc=f"{task} → LeRobot"):
        frames, success = replay_and_record(env, demo)
        if not success:
            skipped.append(demo.key)
            continue
        for f in frames:
            frame = {
                "observation.state": f["obs"]["observation.state"],
                "action": f[action_space],
                "task": spec.language,
            }
            for name in CAMERAS:
                frame[f"observation.images.{name}"] = f["obs"][
                    f"observation.images.{name}"
                ]
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    env.close()

    print(f"Saved {ds.num_episodes} episodes / {ds.num_frames} frames to {root}")
    if skipped:
        print(f"Skipped {len(skipped)} demos whose replay failed: {skipped}")
    return ds


def main():
    parser = argparse.ArgumentParser(description="robomimic → LeRobot v3")
    parser.add_argument("--task", default="lift")
    parser.add_argument("--quality", default="ph", help="ph (proficient human) or mh")
    parser.add_argument("--root", required=True, help="Output dataset directory")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--action_space", choices=["abs", "delta"], default="abs")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(
        args.task,
        args.root,
        repo_id=args.repo_id,
        quality=args.quality,
        image_size=args.image_size,
        action_space=args.action_space,
        max_demos=args.max_demos,
        overwrite=args.overwrite,
        vcodec=args.vcodec,
    )


if __name__ == "__main__":
    main()
