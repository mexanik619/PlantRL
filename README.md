# PlantRL

RL-based plant growth simulation using Soft Actor-Critic (SAC). A plant grows in continuous 2D space, responding to light (phototropism), water (hydrotropism), and gravity (gravitropism) via a sequential multi-agent system with shared policy weights.

## Project Structure

```
PlantRL/
├── pyproject.toml                # package metadata, dependencies, pytest config, CLI entry point
├── requirements.txt              # `-e .[viz,dev]` (dependencies live in pyproject.toml)
├── src/plantrl/
│   ├── agents/
│   │   ├── types.py              # Shared data: StopReason, AgentHandoff, WorldSnapshot, GlobalPlantState
│   │   ├── base_agent.py         # 28-float obs builder, growth step, reward, terminal hooks
│   │   ├── primary_root_agent.py # Strong gravitropism↓ + hydrotropism
│   │   ├── primary_shoot_agent.py# Strong phototropism↑ + gravitropism
│   │   ├── lateral_root_agent.py # Hydrotropism, shorter budget
│   │   ├── branch_agent.py       # Phototropism, auxin suppression (apical dominance)
│   │   └── agent_registry.py     # Maps agent_type string → class
│   ├── world/
│   │   ├── light_field.py        # 1/(1+k·d²) intensity falloff, optional moving source
│   │   ├── moisture_field.py     # Gaussian pockets + ∇M via finite differences
│   │   └── hormone_field.py      # Auxin/cytokinin on 32×32 coarse grid + bilinear interp
│   ├── environment/
│   │   └── env.py                # PlantEnv — Gymnasium env with FIFO queue + curriculum
│   ├── viz/                      # recorder (gym wrapper → .npz) + matplotlib replay renderer
│   ├── visualize.py              # `plantrl-viz` CLI: record / replay episodes
│   └── train.py                  # SAC training + 3-stage curriculum callbacks
├── tests/
│   ├── test_world.py
│   ├── test_env.py
│   └── test_viz.py
└── (git-ignored, generated)  checkpoints/  logs/  runs/
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # editable install of `plantrl` + matplotlib + pytest

python -m pytest -v                      # run tests

# Smoke test (single episode, no training)
python -c "
from plantrl.environment import PlantEnv
env = PlantEnv(stage=1)
obs, _ = env.reset(seed=0)
print('obs shape:', obs.shape)
done = False; steps = 0
while not done:
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    done = term or trunc; steps += 1
print(f'Episode done in {steps} steps. Final reward: {r:.3f}')
"

# Train (stage 1 → auto-advances through the curriculum)
plantrl-train --stage 1 --total-steps 1000000     # or: python -m plantrl.train ...
tensorboard --logdir logs
```

## Design Overview

### Multi-Agent System
Sequential FIFO queue — one meristem tip active at a time. All agents share one SAC policy; behavioural divergence is driven by observation context (`is_root`, `branch_depth`, hormone levels) not separate networks.

### Observation (28 floats)
| Slice | Content |
|-------|---------|
| [0:8] | Tropism signals: light dir/intensity, moisture grad/level, gravity |
| [8:14] | Geometry: sin/cos(tip_angle), sin/cos(parent_angle), normalised pos |
| [14:16] | Hormonal: auxin_local, cytokinin_local |
| [16:19] | Identity: is_root, branch_depth/4, budget_fraction |
| [19:23] | Handoff: one-hot stop_reason from parent |
| [23:28] | Global: shoot_light_deficit, root_water_found, biomass, height, depth |

### Curriculum Stages
| Stage | Condition |
|-------|-----------|
| 1 | Fixed light, single moisture, no obstacles |
| 2 | Random positions, simple obstacles, ±15° gravity |
| 3 | Moving light, depleting soil, dense obstacles, ±30° gravity mid-episode |

The number of parallel envs is one constant for the whole run (`CURRICULUM[*]["n_envs"]` in `train.py`):
SB3 cannot change it mid-run without rebuilding the model.

### Reward design

Per-step reward = `photo + hydro + gravi + progress + cost`, plus `GOAL_BONUS` (10) when an agent
actually **finds** water or light. Hitting the depth/height cap ends the agent but pays nothing.

| Term | Meaning |
|------|---------|
| `progress` | `W_PROGRESS` × world units moved closer to the target (nearest live water pocket for roots, light source for shoots/branches). Uses the true target position in the reward only; it is never part of the observation. |
| `hydro`, `photo` | alignment of the growth direction with the moisture / light gradient |
| `gravi` | alignment with gravity: weight 0.1 for roots, 0 for shoots/branches (a positive weight there would penalise growing up) |
| `cost` | energy cost per step, collision penalty |

Gravity must stay a *mild* bias. With large weights, growing straight down scores higher than steering to
the water, and nothing can be learned. `tests/test_reward_alignment.py` guards this (a hand-coded steerer must
beat a policy that never steers); run it after every reward change.

### Adding a New Agent Type
1. Create `src/plantrl/agents/my_agent.py` subclassing `BaseAgent`
2. Add one line to `REGISTRY` in `agent_registry.py`

`environment/env.py` never imports agent classes directly — only uses the registry.

## Visualizer

Record one episode (any policy) and replay it offline — rendering never runs inside the training loop.
Needs the `viz` extra (matplotlib; `ffmpeg` only for `.mp4`).

```bash
# "do nothing" baseline vs random vs a trained checkpoint (same --seed => same world)
plantrl-viz --stage 1 --policy zero   --seed 0 --out runs/viz/zero_s1.mp4
plantrl-viz --stage 1 --policy random --seed 0 --out runs/viz/random_s1.gif
plantrl-viz --stage 1 --policy checkpoint --model checkpoints/best/best_model.zip --seed 0 --out runs/viz/best_s1.mp4

# single frame / hormone layers / re-render a saved recording (no env or model needed)
plantrl-viz --stage 3 --policy random --out ep.png --frame 60 --fields moisture light auxin cyto
plantrl-viz --replay runs/viz/random_s1.npz --out again.mp4
```

Panels: the world (moisture / light / hormone layers, obstacles, plant coloured by agent type, handoff and
spawn markers), step info, **cumulative reward split by term** (photo / hydro / gravi / cost / goal) and the
28-float observation grouped by slice.
