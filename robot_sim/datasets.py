"""
robomimic v1.5 dataset access — download and iterate demonstrations.

Each demo in demo_v15.hdf5 holds the MuJoCo model XML ('model_file' attr), the
flattened simulator state at every step ('states') and the robosuite-native delta
actions ('actions', gripper -1 = open / +1 = close). Images are not stored; they
are rendered by robot_sim.convert at whatever resolution/cameras we choose.
"""

import json
import os
import urllib.request
from dataclasses import dataclass

import h5py
import numpy as np

from robot_sim.tasks import get_task

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "robomimic"
)


@dataclass
class Demo:
    key: str
    model_xml: str
    states: np.ndarray  # (T, state_dim)
    actions: np.ndarray  # (T, 7) robosuite-native delta actions


def dataset_path(task, quality="ph", root=DEFAULT_ROOT):
    return os.path.join(root, task, quality, "demo_v15.hdf5")


def download(task, quality="ph", root=DEFAULT_ROOT):
    """Download the robomimic file if missing; returns its local path."""
    path = dataset_path(task, quality, root)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = get_task(task).dataset_url(quality)
        print(f"Downloading {url}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def env_args(path):
    with h5py.File(path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def demo_keys(path, split=None):
    """Demo keys in numeric order; split='train'/'valid' uses the file's official mask."""
    with h5py.File(path, "r") as f:
        if split is None:
            keys = list(f["data"].keys())
        else:
            keys = [k.decode() for k in f["mask"][split][:]]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def iter_demos(path, keys=None):
    with h5py.File(path, "r") as f:
        for key in keys or demo_keys(path):
            g = f["data"][key]
            yield Demo(
                key=key,
                model_xml=g.attrs["model_file"],
                states=g["states"][:],
                actions=g["actions"][:],
            )
