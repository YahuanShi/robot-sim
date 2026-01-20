"""
Closed-loop evaluation: run a policy for N seeded episodes and report the success rate.

Episodes use seeds start_seed, start_seed + 1, ... so two policies evaluated with the
same arguments face exactly the same initial object placements. robomimic's demos
were collected on other placements, so the default start seed (100_000) keeps the
evaluation away from any accidental overlap with training data.

Usage (from a policy repo, with its own Policy adapter):
    from robot_sim.evaluate import evaluate
    result = evaluate(policy, task="lift", n_episodes=50, video_dir="outputs/videos")
"""

import json
import os
import time

import imageio
import numpy as np

from robot_sim.envs import FrankaSimEnv


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def run_episode(env, policy, seed, record=False):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    frames = [env.render()] if record else []
    success, steps = False, 0
    while True:
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        steps += 1
        if record:
            frames.append(env.render())
        if info["is_success"]:
            success = True
        if terminated or truncated:
            break
    return {"seed": seed, "success": success, "steps": steps}, frames


def evaluate(
    policy,
    task="lift",
    n_episodes=50,
    start_seed=100_000,
    image_size=84,
    action_space="abs",
    max_steps=None,
    video_dir=None,
    n_videos=5,
    output_json=None,
    env=None,
):
    """
    Returns {"success_rate", "ci95", "successes", "n_episodes", "mean_steps", "episodes"}.

    Videos of the first n_videos episodes are written to video_dir as
    seed<seed>_<success|fail>.mp4 (agentview, render_size of the env).
    """
    own_env = env is None
    if own_env:
        env = FrankaSimEnv(
            task, image_size=image_size, action_space=action_space, max_steps=max_steps
        )
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    episodes = []
    t0 = time.time()
    for i in range(n_episodes):
        seed = start_seed + i
        record = video_dir is not None and i < n_videos
        ep, frames = run_episode(env, policy, seed, record)
        episodes.append(ep)
        if record:
            tag = "success" if ep["success"] else "fail"
            imageio.mimsave(
                os.path.join(video_dir, f"seed{seed}_{tag}.mp4"), frames, fps=20
            )
        done = sum(e["success"] for e in episodes)
        print(
            f"  episode {i + 1}/{n_episodes} seed={seed} success={ep['success']} "
            f"steps={ep['steps']}  ({done}/{i + 1} so far)"
        )
    if own_env:
        env.close()

    k = sum(e["success"] for e in episodes)
    result = {
        "task": task,
        "action_space": action_space,
        "success_rate": k / n_episodes,
        "ci95": wilson_interval(k, n_episodes),
        "successes": k,
        "n_episodes": n_episodes,
        "start_seed": start_seed,
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "wall_time_s": time.time() - t0,
        "episodes": episodes,
    }
    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
    lo, hi = result["ci95"]
    print(
        f"{task}: success {k}/{n_episodes} = {k / n_episodes:.0%} (95% CI {lo:.0%}–{hi:.0%})"
    )
    return result
