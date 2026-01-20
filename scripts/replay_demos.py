"""
Replay robomimic demonstrations open-loop and check they still succeed.

This is the compatibility gate between the dataset and the installed robosuite /
MuJoCo versions: if recorded actions no longer reproduce a success from the recorded
initial state, policies trained on the data cannot be evaluated meaningfully here.

Also reports how far the replayed positions (MuJoCo qpos: joints + object poses) drift
from the recorded ones. Small drift is expected — contact physics differs slightly
between MuJoCo versions — what matters is that the task still succeeds.

Usage:
    python scripts/replay_demos.py --task lift --n 20
"""

import argparse

import numpy as np
from tqdm import tqdm

from robot_sim.datasets import demo_keys, download, iter_demos
from robot_sim.envs import FrankaSimEnv


def active_qpos_mask(env, state):
    """
    qpos entries that matter: everything except objects parked far outside the scene.

    PickPlace tasks keep their unused objects (bread, milk, cereal) at ~(10, 10, 10),
    where they free-fall differently in every replay — irrelevant to the task.
    """
    import mujoco

    model = env._env.sim.model._model
    qpos = state[1 : 1 + env.nq]
    mask = np.ones(env.nq, dtype=bool)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            adr = model.jnt_qposadr[j]
            if np.linalg.norm(qpos[adr : adr + 3]) > 5.0:
                mask[adr : adr + 7] = False
    return mask


def replay(env, demo):
    env.reset_to(demo.model_xml, demo.states[0])
    nq = env.nq
    mask = active_qpos_mask(env, demo.states[0])
    succeeded, drift = False, []
    for t, action in enumerate(demo.actions):
        env.step_sim(action)
        succeeded |= env.is_success()
        if t + 1 < len(demo.states):
            # flattened state = [time, qpos (nq), qvel (nv)]
            qpos, rec_qpos = (
                env.get_sim_state()[1 : 1 + nq],
                demo.states[t + 1][1 : 1 + nq],
            )
            drift.append(np.abs(qpos - rec_qpos)[mask].max())
    return succeeded, float(np.max(drift)) if drift else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="lift")
    parser.add_argument("--quality", default="ph")
    parser.add_argument("--n", type=int, default=20, help="Number of demos to replay")
    args = parser.parse_args()

    path = download(args.task, args.quality)
    keys = demo_keys(path)[: args.n]
    env = FrankaSimEnv(args.task, action_space="delta", image_size=32)
    results = [replay(env, d) for d in tqdm(iter_demos(path, keys), total=len(keys))]
    env.close()

    ok = sum(s for s, _ in results)
    drifts = np.array([d for _, d in results])
    print(f"{args.task}: {ok}/{len(results)} demos succeed on open-loop replay")
    print(f"max qpos drift: median {np.median(drifts):.4f}, worst {drifts.max():.4f}")
    failed = [k for k, (s, _) in zip(keys, results, strict=True) if not s]
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
