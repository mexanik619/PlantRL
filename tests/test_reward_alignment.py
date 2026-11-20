"""Reward-alignment tests: the reward must pay for what we want the plant to do.

The failure these guard against: a reward where growing straight (doing nothing) scores
higher than steering toward water/light, so no amount of training can learn the task.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plantrl.agents.primary_root_agent import PrimaryRootAgent
from plantrl.agents.primary_shoot_agent import PrimaryShootAgent
from plantrl.agents.types import StopReason, WorldSnapshot
from plantrl.environment.env import PlantEnv


def _snap(**kw) -> WorldSnapshot:
    base = dict(
        light_dir_x=0.0, light_dir_y=1.0, light_intensity=0.0,
        moisture_grad_x=0.0, moisture_grad_y=0.0, moisture_level=0.0,
        gravity_x=0.0, gravity_y=-1.0, auxin_local=0.0, cytokinin_local=0.0,
    )
    base.update(kw)
    return WorldSnapshot(**base)


# ── per-step reward terms ────────────────────────────────────────────────

def test_progress_rewards_moving_toward_the_target():
    snap = _snap(water_target=(0.0, -10.0))
    toward = PrimaryRootAgent(start_pos=(0.0, -1.0))      # heads straight down
    toward.step(0.0, snap)
    away = PrimaryRootAgent(start_pos=(0.0, -1.0))
    away.tip_angle = math.pi / 2                          # heads straight up
    away.step(0.0, snap)
    step = toward.STEP_LENGTH * toward.W_PROGRESS
    assert toward.last_reward_terms["progress"] == pytest.approx(+step, abs=1e-6)
    assert away.last_reward_terms["progress"] == pytest.approx(-step, abs=1e-6)


def test_progress_is_zero_without_target_or_on_collision():
    a = PrimaryRootAgent(start_pos=(0.0, -1.0))
    a.step(0.0, _snap())                                  # no water_target
    assert a.last_reward_terms["progress"] == 0.0
    b = PrimaryRootAgent(start_pos=(0.0, -1.0))
    b.step(0.0, _snap(water_target=(0.0, -10.0)), obstacles=[(-1.0, -1.5, 2.0, 1.0)])
    assert b.last_reward_terms["progress"] == 0.0         # tip did not move


def test_goal_bonus_requires_a_real_find():
    # Growing straight into the depth cap ends the agent but must NOT pay the bonus.
    cap = PrimaryRootAgent(start_pos=(0.0, -9.9))
    _, r_cap, done, reason = cap.step(0.0, _snap())
    assert done and reason == StopReason.BUDGET
    assert r_cap < cap.GOAL_BONUS / 2
    # Actually finding water does.
    find = PrimaryRootAgent(start_pos=(0.0, -5.0))
    _, r_find, done, reason = find.step(0.0, _snap(moisture_level=0.95))
    assert done and reason == StopReason.WATER_FOUND
    assert r_find > find.GOAL_BONUS - 1.0


def test_gravity_term_does_not_penalise_shoots_for_growing_up():
    shoot = PrimaryShootAgent(start_pos=(0.0, 1.0))
    shoot.step(0.0, _snap(light_target=(0.0, 10.0)))
    assert shoot.last_reward_terms["gravi"] >= 0.0


# ── whole-episode alignment ──────────────────────────────────────────────

def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _episode_return(kind: str, seed: int, stage: int = 1) -> float:
    """Return of an episode under a policy that never steers ('zero') or a hand-coded
    steerer ('oracle') that turns toward the water (roots) / light (shoots)."""
    env = PlantEnv(stage=stage)
    env.reset(seed=seed)
    u = env.unwrapped
    total, done = 0.0, False
    while not done:
        ag = u.active_agent
        if kind == "zero":
            action = np.zeros(1, np.float32)
        else:
            if ag.IS_ROOT:
                s = max(u.moisture.sources, key=lambda s: s.peak)
                tx, ty = s.cx, s.cy
            else:
                tx, ty = u.light.source_x, u.light.source_y
            err = _wrap(math.atan2(ty - ag.tip_pos[1], tx - ag.tip_pos[0]) - ag.tip_angle)
            action = np.array([np.clip(err, -0.35, 0.35)], np.float32)
        _, r, term, trunc, _ = env.step(action)
        total += r
        done = term or trunc
    return total


def test_steering_toward_targets_beats_doing_nothing_in_stage_1():
    # On the original reward this gap was about -14 (steering LOST); it must be clearly positive.
    gaps = [_episode_return("oracle", s) - _episode_return("zero", s) for s in range(10)]
    assert np.mean(gaps) > 10.0, f"mean paired gap {np.mean(gaps):.1f}"
    assert sum(g > 0 for g in gaps) >= 8, f"oracle only beat zero on {sum(g > 0 for g in gaps)}/10 seeds"
