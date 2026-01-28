"""
Minimal policy interface shared by every policy evaluated in robot_sim.

A policy sees one observation per control step and returns one action. Anything
policy-specific — observation history (Diffusion Policy), action chunk queues,
temporal ensembling (ACT) — lives inside the policy object, so the evaluation loop
is identical for every method and results are directly comparable.
"""

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def reset(self, seed: int | None = None) -> None:
        """Called at the start of every episode (clear history / queues).

        seed is the episode's environment seed. Stochastic policies (e.g. diffusion
        sampling) should seed their own RNG with it, so every episode is reproducible
        on its own, independent of which episodes ran before it.
        """

    def act(self, obs: dict) -> np.ndarray:
        """obs: dict as returned by FrankaSimEnv (HWC uint8 images, float32 state)."""


class ReplayPolicy:
    """Plays back a fixed action sequence — used to test the evaluation loop."""

    def __init__(self, actions):
        self.actions = np.asarray(actions)
        self.t = 0

    def reset(self, seed=None):
        self.t = 0

    def act(self, obs):
        a = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        return a
