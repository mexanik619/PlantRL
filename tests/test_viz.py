"""Tests for the recorder + renderer (viz package)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from plantrl.agents.base_agent import OBS_DIM
from plantrl.environment.env import PlantEnv
from plantrl.viz.recorder import TERM_NAMES, TrajectoryRecorder, load_episode
from plantrl.viz.render import render_episode


def _run(stage: int, seed: int = 0, record_fields: bool = True) -> TrajectoryRecorder:
    env = TrajectoryRecorder(PlantEnv(stage=stage), record_fields=record_fields)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
    return env


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_recording_shapes_and_consistency(stage):
    rec = _run(stage)
    ep = rec.episode()
    T = len(ep["reward"])
    assert T > 0
    assert ep["obs"].shape == (T, OBS_DIM)
    assert ep["start"].shape == ep["end"].shape == (T, 2)
    assert ep["terms"].shape == (T, len(TERM_NAMES))
    # reward terms must add up to the reward the env returned
    np.testing.assert_allclose(ep["terms"].sum(axis=1), ep["reward"], atol=1e-4)
    # a segment is added iff the tip actually moved
    moved = np.linalg.norm(ep["end"] - ep["start"], axis=1) > 1e-6
    assert np.array_equal(moved, ep["seg_added"])
    # the first two agents are the primaries
    assert set(ep["agent_type"][ep["agent_id"] < 2]) == {"primary_root", "primary_shoot"}


def test_npz_roundtrip_and_png_gif(tmp_path):
    rec = _run(1)
    path = rec.save(tmp_path / "ep.npz")
    ep = load_episode(path)
    assert ep["meta"]["stage"] == 1
    assert "auxin" in ep and ep["auxin"].shape[1:] == (32, 32)

    png = tmp_path / "ep.png"
    render_episode(ep, png, fields=("moisture", "light", "auxin", "cyto"))
    assert png.stat().st_size > 5_000

    gif = tmp_path / "ep.gif"
    render_episode(ep, gif, every=20, fps=5, dpi=40)
    assert gif.stat().st_size > 1_000


def test_no_fields_recording_rejects_hormone_layers(tmp_path):
    ep = _run(1, record_fields=False).episode()
    assert "auxin" not in ep
    ep["meta"] = __import__("json").loads(str(ep["meta"]))
    with pytest.raises(ValueError):
        render_episode(ep, tmp_path / "x.png", fields=("auxin",))


def test_recorder_does_not_change_env_behaviour():
    """Same seed + same actions → identical returns with and without the wrapper."""
    def rollout(wrap: bool) -> float:
        env = PlantEnv(stage=2)
        if wrap:
            env = TrajectoryRecorder(env)
        env.reset(seed=3)
        rng = np.random.default_rng(3)
        total, done = 0.0, False
        while not done:
            a = rng.uniform(-0.35, 0.35, size=1).astype(np.float32)
            _, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        return total

    assert rollout(False) == pytest.approx(rollout(True))
