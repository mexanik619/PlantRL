"""RunRecorder: every episode is saved, detail episodes keep grids, summaries are returned."""

from __future__ import annotations

import numpy as np

from plantrl.environment.env import PlantEnv
from plantrl.viz.recorder import load_episode
from plantrl.viz.run_recorder import SUMMARY_KEY, RunRecorder


def test_every_episode_saved_and_detail_flag(tmp_path):
    env = RunRecorder(PlantEnv(stage=1), run_dir=tmp_path, detail_every=2)
    summaries = []
    for ep_i in range(4):                       # 4 episodes, detail on episodes 0 and 2
        env.reset(seed=ep_i)
        done = False
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        summaries.append(info[SUMMARY_KEY])
    files = sorted((tmp_path / "episodes").glob("*.npz"))
    assert len(files) == 4
    assert [s["detail"] for s in summaries] == [True, False, True, False]
    assert [s["episode"] for s in summaries] == [0, 1, 2, 3]
    detail = load_episode(tmp_path / summaries[0]["trajectory_file"])
    plain = load_episode(tmp_path / summaries[1]["trajectory_file"])
    assert "auxin" in detail and detail["auxin"].shape[1:] == (32, 32)
    assert "auxin" not in plain
    # the summary matches the saved episode
    assert summaries[1]["steps"] == len(plain["reward"])
    np.testing.assert_allclose(summaries[1]["return"], float(plain["reward"].sum()), rtol=1e-5)


def test_stage_switch_passes_through_the_wrapper(tmp_path):
    env = RunRecorder(PlantEnv(stage=1), run_dir=tmp_path)
    env.unwrapped.set_stage(3)
    env.reset(seed=0)
    assert len(env.unwrapped._obstacles) >= 4