"""Smoke tests for PlantEnv Gymnasium environment."""

from __future__ import annotations


import numpy as np
import pytest

from plantrl.environment.env import PlantEnv
from plantrl.agents.base_agent import OBS_DIM


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(params=[1, 2, 3])
def env(request):
    """PlantEnv instantiated for each curriculum stage."""
    e = PlantEnv(stage=request.param, seed=0)
    yield e
    e.close()


# ──────────────────────────────────────────────
# Reset
# ──────────────────────────────────────────────

class TestReset:
    def test_obs_shape(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_obs_dtype(self, env):
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32

    def test_obs_bounded(self, env):
        obs, _ = env.reset(seed=42)
        # obs should be in roughly [-1, 1] range (some dims may slightly exceed)
        assert np.all(np.isfinite(obs)), "obs contains NaN or Inf"

    def test_reset_is_reproducible(self):
        e = PlantEnv(stage=1, seed=7)
        obs1, _ = e.reset(seed=7)
        obs2, _ = e.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)
        e.close()


# ──────────────────────────────────────────────
# Step
# ──────────────────────────────────────────────

class TestStep:
    def test_step_output_types(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, *_ = env.step(action)
        assert obs.shape == (OBS_DIM,)

    def test_action_space_bounds(self, env):
        env.reset(seed=0)
        # Actions at boundaries should not crash
        for action_val in [-0.35, 0.0, 0.35]:
            obs, r, term, trunc, _ = env.step(np.array([action_val], dtype=np.float32))
            assert np.all(np.isfinite(obs))
            if term or trunc:
                break


# ──────────────────────────────────────────────
# Full episode
# ──────────────────────────────────────────────

class TestFullEpisode:
    def _run_episode(self, env, seed=0):
        obs, _ = env.reset(seed=seed)
        steps = 0
        total_reward = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            assert np.all(np.isfinite(obs)), f"NaN/Inf obs at step {steps}"
            if terminated or truncated:
                break
            if steps > 5000:
                raise RuntimeError("Episode did not terminate within 5000 steps")
        return steps, total_reward

    def test_stage1_episode_completes(self):
        env = PlantEnv(stage=1, seed=0)
        steps, reward = self._run_episode(env)
        assert steps > 0
        env.close()

    def test_stage2_episode_completes(self):
        env = PlantEnv(stage=2, seed=1)
        steps, _ = self._run_episode(env)
        assert steps > 0
        env.close()

    def test_stage3_episode_completes(self):
        env = PlantEnv(stage=3, seed=2)
        steps, _ = self._run_episode(env, seed=2)
        assert steps > 0
        env.close()

    def test_multiple_resets_stable(self):
        env = PlantEnv(stage=1, seed=0)
        for i in range(5):
            self._run_episode(env, seed=i)
        env.close()


# ──────────────────────────────────────────────
# Action space
# ──────────────────────────────────────────────

class TestActionSpace:
    def test_action_space_shape(self):
        env = PlantEnv(stage=1)
        assert env.action_space.shape == (1,)
        env.close()

    def test_action_space_bounds(self):
        env = PlantEnv(stage=1)
        assert float(env.action_space.low[0]) == pytest.approx(-0.35)
        assert float(env.action_space.high[0]) == pytest.approx(0.35)
        env.close()

    def test_observation_space_shape(self):
        env = PlantEnv(stage=1)
        assert env.observation_space.shape == (OBS_DIM,)
        env.close()
