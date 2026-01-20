"""
Franka Panda simulation environment (robosuite) with a LeRobot-style interface.

Observation (dict, matches the LeRobot v3 datasets written by robot_sim.convert):
  observation.images.agentview  (H, W, 3) uint8  — fixed front camera  (≈ real "exterior")
  observation.images.wrist      (H, W, 3) uint8  — eye-in-hand camera  (≈ real "wrist")
  observation.state             (9,) float32     — eef pos (3), eef site quat xyzw (4), finger qpos (2)
                                                   (same frame as absolute actions)

Action (float32):
  action_space="abs"   → (10,) [x, y, z, r6d (6), gripper]: absolute eef target in the world frame,
                         position in m, orientation in the continuous 6D representation
                         (first two rows of the rotation matrix, see robot_sim.rotations)
  action_space="delta" → (7,) [dx, dy, dz, drx, dry, drz, gripper]: robomimic's native delta in
                         [-1, 1], scaled by the controller to ±5 cm / ±0.5 rad per step
  gripper: 1 = open, 0 = closed — the same convention as the real-robot datasets.
           robosuite itself uses -1 = open, +1 = close; conversion happens here only.
"""

import os

os.environ.setdefault(
    "MUJOCO_GL", "egl"
)  # headless rendering; must precede mujoco import

import gymnasium as gym
import numpy as np
import robosuite
import robosuite.macros as macros

from robot_sim.rotations import rot6d_to_rotvec
from robot_sim.tasks import controller_config, get_task

# Observations come out upright (row 0 = top) instead of OpenGL's bottom-up order
macros.IMAGE_CONVENTION = "opencv"

CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
# eef_quat_site is the orientation the OSC controller tracks; robosuite keeps the legacy
# "robot0_eef_quat" (gripper body frame, ~90° off) only for old datasets. Using the site
# keeps the state and absolute actions in the same frame.
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat_site", "robot0_gripper_qpos")
STATE_DIM = 9
ACTION_DIM = {"abs": 10, "delta": 7}


def gripper_to_sim(g):
    """Dataset convention (1 = open, 0 = closed) → robosuite (-1 = open, +1 = close)."""
    return 1.0 - 2.0 * np.clip(g, 0.0, 1.0)


def gripper_from_sim(a):
    """robosuite (-1 = open, +1 = close) → dataset convention (1 = open, 0 = closed)."""
    return (1.0 - np.clip(a, -1.0, 1.0)) / 2.0


class FrankaSimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task="lift",
        image_size=84,
        action_space="abs",
        max_steps=None,
        render_size=256,
    ):
        self.task = get_task(task)
        self.image_size = image_size
        self.action_space_type = action_space
        self.max_steps = max_steps or self.task.max_steps
        self.render_size = render_size
        self.render_mode = "rgb_array"

        self._env = robosuite.make(
            self.task.env_name,
            robots="Panda",
            controller_configs=controller_config(action_space),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=list(CAMERAS.values()),
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=False,
            control_freq=20,
            reward_shaping=False,
            ignore_done=True,  # episode length is handled here (max_steps)
            hard_reset=False,  # reuse the MuJoCo model between resets (much faster)
            **self.task.env_kwargs,
        )
        self._t = 0

        img = gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.images.agentview": img,
                "observation.images.wrist": img,
                "observation.state": gym.spaces.Box(
                    -np.inf, np.inf, (STATE_DIM,), np.float32
                ),
            }
        )
        if action_space == "abs":
            low = np.array([-np.inf] * 9 + [0.0], np.float32)
            high = np.array([np.inf] * 9 + [1.0], np.float32)
        else:
            low = np.array([-1.0] * 6 + [0.0], np.float32)
            high = np.array([1.0] * 6 + [1.0], np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # robosuite's object placement samplers draw from the global numpy RNG
            np.random.seed(seed)
        raw = self._env.reset()
        self._t = 0
        return self._obs(raw), {"is_success": False}

    def step(self, action):
        return self._step_sim(self.to_sim_action(action))

    def to_sim_action(self, action):
        """Policy action (this module's convention) → robosuite-native 7-dim action."""
        action = np.asarray(action, dtype=np.float64)
        expected = ACTION_DIM[self.action_space_type]
        if action.shape != (expected,):
            raise ValueError(f"expected action shape ({expected},), got {action.shape}")
        if self.action_space_type == "abs":
            arm = np.concatenate([action[:3], rot6d_to_rotvec(action[3:9])])
        else:
            arm = action[:6]
        return np.concatenate([arm, [gripper_to_sim(action[-1])]])

    def render(self):
        """Higher-resolution agentview frame for videos (rgb, upright)."""
        frame = self._env.sim.render(
            width=self.render_size, height=self.render_size, camera_name="agentview"
        )
        return frame[
            ::-1
        ].copy()  # sim.render is bottom-up regardless of IMAGE_CONVENTION

    def close(self):
        self._env.close()

    # ── Dataset tooling ──────────────────────────────────────────────────────

    def step_sim(self, sim_action):
        """Step with a robosuite-native action (gripper -1 = open), e.g. from robomimic files."""
        return self._step_sim(np.asarray(sim_action, dtype=np.float64))

    def reset_to(self, model_xml, state):
        """Restore a recorded scene (robomimic 'model_file' attr + flattened MuJoCo state)."""
        self._env.reset()
        self._env.reset_from_xml_string(self._env.edit_model_xml(model_xml))
        self._env.sim.reset()
        self._env.sim.set_state_from_flattened(state)
        self._env.sim.forward()
        self._t = 0
        return self._obs(self._env._get_observations(force_update=True))

    def get_sim_state(self):
        """Flattened MuJoCo state: [time, qpos (nq), qvel (nv)]."""
        return self._env.sim.get_state().flatten()

    @property
    def nq(self):
        return self._env.sim.model.nq

    def get_obs(self):
        """Current observation without stepping."""
        return self._obs(self._env._get_observations(force_update=True))

    @property
    def arm_controller(self):
        return self._env.robots[0].part_controllers["right"]

    def is_success(self):
        return bool(self._env._check_success())

    # ── Internals ────────────────────────────────────────────────────────────

    def _step_sim(self, sim_action):
        raw, reward, _, info = self._env.step(sim_action)
        self._t += 1
        success = self.is_success()
        info = dict(info, is_success=success)
        terminated = success
        truncated = self._t >= self.max_steps
        return self._obs(raw), float(reward), terminated, truncated, info

    def _obs(self, raw):
        state = np.concatenate([raw[k] for k in STATE_KEYS]).astype(np.float32)
        obs = {"observation.state": state}
        for name, cam in CAMERAS.items():
            obs[f"observation.images.{name}"] = np.ascontiguousarray(
                raw[f"{cam}_image"]
            )
        return obs


def make_env(task="lift", **kwargs):
    return FrankaSimEnv(task, **kwargs)
