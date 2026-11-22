"""Record a whole training run: every episode, plus a log row per episode.

Written under ``run_dir``::

    episodes.csv                              one row per finished episode, whole run
    episodes/s{stage}_{env}_{n:06d}.npz       trajectory of EVERY episode (no hormone grids)
    episodes/detail_s{stage}_{env}_{n:06d}.npz  every ``detail_every``-th episode per env, WITH grids

Replay any episode with:  plantrl-viz --replay run_dir/episodes/<file>.npz --out ep.mp4
(Detail files are the only ones that can show the auxin / cytokinin layers.)

Size, for orientation: a trajectory without grids is roughly 130 steps x 250 bytes,
so tens of KB per episode and a few hundred MB for 1M steps. Each detail episode adds
two 32x32 float16 grids per step; keep ``detail_every`` large.

Use ``RunRecorder`` only for training envs, never for the eval env.
"""

from __future__ import annotations

import csv
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from plantrl.viz.recorder import TERM_NAMES, TrajectoryRecorder

SUMMARY_KEY = "plant_summary"   # key in the step info dict that carries an episode summary
CSV_FIELDS = [
    "timestep", "env_index", "episode", "stage", "steps", "agents", "return",
    *TERM_NAMES, "stop_reasons", "detail", "trajectory_file",
]


class RunRecorder(TrajectoryRecorder):
    """Gym wrapper that saves each finished episode of a training env.

    The wrapper stays inside each worker process, so episodes are written where
    they happen. The summary also travels back in ``info[SUMMARY_KEY]``, which
    ``RunLogCallback`` turns into a CSV row.
    """

    def __init__(self, env: gym.Env, run_dir: str | Path, detail_every: int = 100) -> None:
        super().__init__(env, record_fields=False)
        self.run_dir = Path(run_dir)
        self.ep_dir = self.run_dir / "episodes"
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.detail_every = max(1, int(detail_every))
        self._env_tag = uuid.uuid4().hex[:8]      # unique per env, even when envs share a process
        self._n_episodes = 0

    def reset(self, **kwargs):
        # The episode about to start is a "detail" episode if its index is a multiple of detail_every.
        self.record_fields = self._n_episodes % self.detail_every == 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            summary = self._finish_episode()
            if summary is not None:
                info = dict(info)
                info[SUMMARY_KEY] = summary
        return obs, reward, terminated, truncated, info

    def _finish_episode(self) -> dict[str, Any] | None:
        if not self._rows:
            return None
        ep = self.episode()
        stage = int(json.loads(str(ep["meta"]))["stage"])
        detail = bool(self.record_fields)
        stem = f"s{stage}_{self._env_tag}_{self._n_episodes:06d}.npz"
        path = self.ep_dir / (("detail_" if detail else "") + stem)
        np.savez_compressed(path, **ep)

        term_sums = ep["terms"].sum(axis=0)
        summary = {
            "episode": self._n_episodes,
            "stage": stage,
            "steps": int(len(ep["reward"])),
            "agents": int(ep["agent_id"].max()) + 1,
            "return": float(ep["reward"].sum()),
            "terms": {k: float(v) for k, v in zip(TERM_NAMES, term_sums)},
            "stop_reasons": dict(Counter(s for s in ep["stop_reason"].tolist() if s)),
            "detail": detail,
            "trajectory_file": str(path.relative_to(self.run_dir)),
        }
        self._n_episodes += 1
        return summary


class RunLogCallback(BaseCallback):
    """Writes one CSV row per finished episode (from RunRecorder summaries)."""

    def __init__(self, run_dir: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self._file = None
        self._writer: csv.DictWriter | None = None

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "episodes.csv"
        is_new = not path.exists() or path.stat().st_size == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if is_new:
            self._writer.writeheader()

    def _on_step(self) -> bool:
        wrote = False
        for env_index, info in enumerate(self.locals.get("infos", [])):
            s = info.get(SUMMARY_KEY)
            if s is None:
                continue
            row = {
                "timestep": int(self.num_timesteps),
                "env_index": env_index,
                "episode": s["episode"],
                "stage": s["stage"],
                "steps": s["steps"],
                "agents": s["agents"],
                "return": round(s["return"], 4),
                **{k: round(v, 4) for k, v in s["terms"].items()},
                "stop_reasons": json.dumps(s["stop_reasons"]),
                "detail": int(s["detail"]),
                "trajectory_file": s["trajectory_file"],
            }
            self._writer.writerow(row)
            wrote = True
        if wrote:
            self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None