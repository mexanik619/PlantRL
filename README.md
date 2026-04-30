# PlantRL

RL-based plant growth simulation using Soft Actor-Critic (SAC). A plant grows in continuous 2D space, responding to light (phototropism), water (hydrotropism), and gravity (gravitropism) via a sequential multi-agent system with shared policy weights.

## Project Structure

```
PlantRL/
├── src/
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
│   └── train.py                  # SAC training + 3-stage curriculum callbacks
├── tests/
│   ├── test_world.py
│   └── test_env.py
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies (into existing venv)
pip install -r requirements.txt

# Run tests
cd /home/mikocat/Repos/PlantRL
python -m pytest tests/ -v

# Smoke test (single episode, no training)
python -c "
import sys; sys.path.insert(0, 'src')
from environment.env import PlantEnv
env = PlantEnv(stage=1)
obs, _ = env.reset()
print('obs shape:', obs.shape)
done = False; steps = 0
while not done:
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    done = term or trunc; steps += 1
print(f'Episode done in {steps} steps. Final reward: {r:.3f}')
"

# Train (stage 1 → auto-advances through curriculum)
python src/train.py --stage 1 --total-steps 1000000
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
| Stage | Envs | Condition |
|-------|------|-----------|
| 1 | 4 | Fixed light, single moisture, no obstacles |
| 2 | 8 | Random positions, simple obstacles, ±15° gravity |
| 3 | 8–16 | Moving light, depleting soil, dense obstacles, ±30° gravity mid-episode |

### Adding a New Agent Type
1. Create `src/agents/my_agent.py` subclassing `BaseAgent`
2. Add one line to `REGISTRY` in `agent_registry.py`

`plant_env.py` never imports agent classes directly — only uses the registry.
