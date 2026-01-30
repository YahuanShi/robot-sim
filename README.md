# robot-sim

Shared Franka Panda simulation for my imitation-learning reimplementations
([diffusion-policy-reimplementation](../diffusion-policy-reimplementation)).

- **Simulator**: [robosuite](https://robosuite.ai) 1.5 (MuJoCo), Franka Panda + parallel gripper, OSC_POSE control at 20 Hz
- **Data**: [robomimic](https://robomimic.github.io) v1.5 proficient-human demos (200 per task), converted to **LeRobot v3** with rendered images
- **Evaluation**: one closed-loop harness with fixed seeds — every policy is scored on exactly the same initial object placements

| Task | robosuite env | What | Demo length | Usable demos |
|---|---|---|---|---|
| `lift` | `Lift` | lift the cube | 36–64 steps | 200 / 200 |
| `can` | `PickPlaceCan` | pick the can, place it in its bin slot | 82–151 steps | 194 / 200 |
| `square` | `NutAssemblySquare` | pick the square nut, fit it onto the peg | 107–236 steps | 162 / 200 |

---

## Setup

```bash
uv sync --extra convert --extra dev        # own venv (.venv), incl. LeRobot for conversion

# Inside a policy repo's environment (editable, so changes here apply immediately):
uv pip install --python DP/bin/python -e ../robot-sim                         # diffusion-policy-reimplementation
```

Rendering is headless via EGL (`MUJOCO_GL=egl` is set automatically; use `osmesa` without a GPU).

**Pinned versions, and why**
- `robosuite==1.5.1` — 1.5.2 pins `mink==0.0.5`, which requires `numpy<2`; LeRobot 0.4 (used by both policy repos) needs `numpy>=2`.
- `mujoco>=3.3,<3.9` — newer MuJoCo breaks robosuite 1.5.1 (joint-type assertion at env creation).
- `lerobot>=0.4,<0.5` — 0.5 requires Python ≥ 3.12; the policy repos are on 3.10.

## Workflow

```bash
# 1. Compatibility gate: do the recorded actions still reproduce the demos here?
python scripts/replay_demos.py --task lift --n 200      # → 200/200
python scripts/replay_demos.py --task can --n 200       # → 194/200 (6 fail and are skipped)
python scripts/replay_demos.py --task square --n 200    # → 162/200 (tight peg fit: open-loop drift)

# 2. Absolute actions reproduce the demos as well as the original deltas?
python scripts/check_abs_actions.py --task can --n 20   # → same successes, final eef within ~0.5 mm

# 3. Convert to LeRobot v3 (downloads the robomimic file on first use, ~1 min per 40 demos)
python -m robot_sim.convert --task lift --root data/lerobot/lift_ph_abs_84
python -m robot_sim.convert --task can  --root data/lerobot/can_ph_abs_84
python -m robot_sim.convert --task square --root data/lerobot/square_ph_abs_84

# 4. Train in a policy repo on the converted dataset, then evaluate there:
#    DP:  python eval.py --checkpoint <ckpt> --device cuda --sim lift --n_episodes 50
```

## Conventions

**Observation** (dict, same keys as the LeRobot datasets)

| Key | Shape | Content |
|---|---|---|
| `observation.images.agentview` | (H, W, 3) uint8 | fixed front camera (≈ real-robot "exterior") |
| `observation.images.wrist` | (H, W, 3) uint8 | eye-in-hand camera |
| `observation.state` | (9,) float32 | eef position (3), eef **site** quaternion xyzw (4), finger joints (2) |

**Action** — `action_space="abs"` (default, recommended) or `"delta"`

| Space | Shape | Content |
|---|---|---|
| `abs` | (10,) | eef target position (3, world frame, m), orientation as **6D rotation** (6), gripper |
| `delta` | (7,) | robomimic's native per-step delta in [-1, 1] (±5 cm / ±0.5 rad), gripper |

Things that are easy to get wrong (each one was checked here):
- **Gripper: 1 = open, 0 = closed** — same as the real-robot datasets. robosuite uses −1 = open / +1 = close; the conversion happens only in `robot_sim.envs`.
- **Orientation frame**: the state uses `robot0_eef_quat_site`, the frame the OSC controller tracks. robosuite's legacy `robot0_eef_quat` is ~90° off; with it, "hold the current pose" as an absolute action moves the arm by 10 cm.
- **6D rotation, not axis-angle**: with the gripper pointing down the rotation is near π, where the axis-angle vector flips sign between neighbouring frames — a terrible regression target. Diffusion Policy converts robomimic absolute actions to 6D for the same reason.
- **Absolute actions** are the OSC controller's goal pose after applying each recorded delta, read while replaying the demo. Images are rendered during the same replay, so images, states and actions describe one consistent trajectory.
- **Seeds**: robosuite places objects with the global NumPy RNG, so `reset(seed)` seeds it; episode *i* of an evaluation uses `start_seed + i`. The same seed goes to `policy.reset(seed)`, so a stochastic policy (Diffusion Policy) reproduces every episode exactly.

## Policy interface

```python
class Policy(Protocol):
    def reset(self, seed=None) -> None: ...      # new episode: clear history / queues;
                                                 # seed = env seed, for stochastic policies
    def act(self, obs: dict) -> np.ndarray: ...  # one observation in, one action out

from robot_sim.evaluate import evaluate
result = evaluate(policy, task="can", n_episodes=50, video_dir="videos/")
# {"success_rate", "ci95", "successes", "episodes": [{"seed", "success", "steps"}, ...], ...}
```

Observation history (Diffusion Policy) and action chunking / temporal ensembling (ACT)
live inside each repo's adapter (`evaluation/sim_policy.py`, `evaluation/franka_sim.py`),
so the evaluation loop is identical for both.

## Layout

```
robot_sim/
├── tasks.py       # task registry + OSC_POSE controller config (copied from the datasets)
├── envs.py        # FrankaSimEnv: Gymnasium env, LeRobot-style obs, abs/delta actions
├── datasets.py    # download + iterate robomimic demo_v15.hdf5
├── convert.py     # replay demos → LeRobot v3 (rendered images, abs or delta actions)
├── rotations.py   # 6D rotation ↔ matrix / axis-angle
├── policy_api.py  # Policy protocol
└── evaluate.py    # seeded closed-loop evaluation, success rate + Wilson CI, videos
scripts/
├── replay_demos.py       # dataset ↔ simulator compatibility check
└── check_abs_actions.py  # absolute-action conversion check
tests/                    # pytest (dataset-dependent tests skip if files are missing)
```
