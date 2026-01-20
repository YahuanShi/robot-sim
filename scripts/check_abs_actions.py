"""
Verify that absolute actions extracted by robot_sim.convert reproduce the demos.

For each demo: replay the recorded delta actions while recording the controller goal
(the absolute action), then replay those absolute actions open-loop in an env with an
absolute-pose controller and check that the task still succeeds.

Usage:
    python scripts/check_abs_actions.py --task lift --n 20
"""

import argparse

import numpy as np
from tqdm import tqdm

from robot_sim.convert import replay_and_record
from robot_sim.datasets import demo_keys, download, iter_demos
from robot_sim.envs import FrankaSimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="lift")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    path = download(args.task)
    keys = demo_keys(path)[: args.n]
    delta_env = FrankaSimEnv(args.task, action_space="delta", image_size=32)
    abs_env = FrankaSimEnv(args.task, action_space="abs", image_size=32)

    n_delta_ok = n_abs_ok = 0
    pos_err = []
    for demo in tqdm(iter_demos(path, keys), total=len(keys)):
        frames, delta_ok = replay_and_record(delta_env, demo)
        n_delta_ok += delta_ok
        delta_final = delta_env.get_obs()["observation.state"][:3]
        obs = abs_env.reset_to(demo.model_xml, demo.states[0])
        abs_ok = False
        for f in frames:
            obs, _, _, _, info = abs_env.step(f["abs"])
            abs_ok |= info["is_success"]
        n_abs_ok += abs_ok
        # final eef position: abs replay vs delta replay of the same demo
        pos_err.append(np.linalg.norm(obs["observation.state"][:3] - delta_final))
    print(
        f"{args.task}: delta replay {n_delta_ok}/{len(keys)}, abs replay {n_abs_ok}/{len(keys)}"
    )
    print(
        f"final eef position, abs vs delta replay: median {np.median(pos_err) * 1000:.1f} mm, "
        f"worst {np.max(pos_err) * 1000:.1f} mm"
    )


if __name__ == "__main__":
    main()
