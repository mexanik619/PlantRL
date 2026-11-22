"""Training-session video and policy demo renderers (small inputs, GIF output)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from plantrl.environment.env import PlantEnv
from plantrl.viz.run_recorder import CSV_FIELDS, SUMMARY_KEY, RunRecorder
from plantrl.viz.run_render import render_demo, render_training_snapshot, render_training_video


def _zero(obs, env):
    return np.zeros(1, dtype=np.float32)


@pytest.fixture()
def run_dir(tmp_path):
    """A tiny finished run: 4 episodes written by RunRecorder + the CSV RunLogCallback writes."""
    rec = RunRecorder(PlantEnv(stage=1), run_dir=tmp_path, detail_every=2)
    rows = []
    for ep_i in range(4):
        rec.reset(seed=ep_i)
        done = False
        while not done:
            _, _, term, trunc, info = rec.step(rec.action_space.sample())
            done = term or trunc
        s = info[SUMMARY_KEY]
        rows.append({
            "timestep": (ep_i + 1) * 100, "env_index": 0, "episode": s["episode"], "stage": s["stage"],
            "steps": s["steps"], "agents": s["agents"], "return": round(s["return"], 4),
            **{k: round(v, 4) for k, v in s["terms"].items()},
            "stop_reasons": json.dumps(s["stop_reasons"]),
            "detail": int(s["detail"]), "trajectory_file": s["trajectory_file"],
        })
    with open(tmp_path / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_training_video_and_snapshot(run_dir, tmp_path):
    out = render_training_video(run_dir, tmp_path / "train.gif", frames=3, fps=2, hold_seconds=0.5, dpi=40)
    assert out.exists() and out.stat().st_size > 1_000
    png = render_training_snapshot(run_dir, tmp_path / "final.png", dpi=40)
    assert png.exists() and png.stat().st_size > 1_000


def test_demo_video_with_baseline(tmp_path):
    out = render_demo(
        [("do nothing", _zero), ("random-ish", lambda obs, env: env.action_space.sample())],
        tmp_path / "demo.gif", stage=1, seeds=[0], fps=2, every=50, card_seconds=0.5, dpi=40,
    )
    assert out.exists() and out.stat().st_size > 1_000