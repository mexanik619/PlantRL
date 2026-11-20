"""TrajectoryRecorder — records one PlantEnv episode so it can be replayed offline.

Design rules:
  * Recording is a plain ``gym.Wrapper``; the env itself never renders and never
    imports matplotlib, so training speed and headless machines are unaffected.
  * Every row describes the world *as the acting agent saw it* (state captured
    before ``env.step``) plus what happened (tip movement, reward terms, handoff).
  * The result is a flat ``.npz`` file — the renderer needs nothing from the env.

Usage:
    env = TrajectoryRecorder(PlantEnv(stage=1))
    obs, _ = env.reset(seed=0)
    ...                                  # run any policy
    env.save("runs/ep0.npz")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# Names for the 28 observation floats (layout fixed in agents/base_agent.py)
OBS_NAMES: list[str] = [
    "light_dx", "light_dy", "light_int", "moist_gx", "moist_gy", "moist_lvl", "grav_x", "grav_y",
    "sin_tip", "cos_tip", "sin_par", "cos_par", "pos_x", "pos_y",
    "auxin", "cytokinin",
    "is_root", "depth/4", "budget_used",
    "stop:water", "stop:light", "stop:budget", "stop:suppr",
    "shoot_light_def", "root_water", "biomass", "height", "depth",
]
# (start, stop, label) slices — used to colour the observation panel
OBS_SLICES: list[tuple[int, int, str]] = [
    (0, 8, "tropism"), (8, 14, "geometry"), (14, 16, "hormonal"),
    (16, 19, "identity"), (19, 23, "handoff"), (23, 28, "global"),
]

TERM_NAMES = ("photo", "hydro", "gravi", "progress", "cost", "goal")


class TrajectoryRecorder(gym.Wrapper):
    """Gymnasium wrapper that logs every step of the *current* episode.

    Args:
        env:           A ``PlantEnv`` (or a wrapper around one).
        record_fields: Also store the 32x32 auxin / cytokinin grids each step
                       (~1 MB per episode as float16). Disable for lean logs.
    """

    def __init__(self, env: gym.Env, record_fields: bool = True) -> None:
        super().__init__(env)
        self.record_fields = record_fields
        self._rows: list[dict[str, Any]] = []
        self._agent_ids: dict[int, int] = {}
        self._meta: dict[str, Any] = {}
        self._last_obs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped
        self._rows = []
        self._agent_ids = {}
        self._last_obs = np.asarray(obs, dtype=np.float32).copy()
        h = u.hormones
        self._meta = {
            "stage": int(u.stage),
            "seed": kwargs.get("seed"),
            "obstacles": [[float(v) for v in o] for o in u._obstacles],
            "moisture": [
                {"cx": float(s.cx), "cy": float(s.cy), "sigma": float(s.sigma)}
                for s in u.moisture.sources
            ],
            "light": {"falloff_k": float(u.light.falloff_k), "intensity": float(u.light.intensity)},
            "bounds": [float(h.x_min), float(h.x_max), float(h.y_min), float(h.y_max)],
            "grid_size": int(h.grid_size),
        }
        return obs, info

    def step(self, action):
        u = self.env.unwrapped
        agent = u.active_agent
        if agent is None or self._last_obs is None:      # nothing to record
            return self.env.step(action)

        # ---- state the agent sees, captured BEFORE the step ----
        aid = self._agent_ids.setdefault(id(agent), len(self._agent_ids))
        start = (float(agent.tip_pos[0]), float(agent.tip_pos[1]))
        n_seg = len(agent.segments)
        obs_before = self._last_obs
        light_xy = (float(u.light.source_x), float(u.light.source_y))
        gravity = (float(u._gravity[0]), float(u._gravity[1]))
        peaks = [float(s.peak) for s in u.moisture.sources]
        auxin = cyto = None
        if self.record_fields:
            auxin = u.hormones._auxin_grid.astype(np.float16).copy()
            cyto = u.hormones._cyto_grid.astype(np.float16).copy()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # ---- what happened ----
        raw = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        lo, hi = float(self.action_space.low[0]), float(self.action_space.high[0])
        terms = dict(getattr(agent, "last_reward_terms", {}))
        terms["goal"] = float(reward) - sum(terms.values()) if terms else 0.0

        self._rows.append({
            "agent_id": aid,
            "agent_type": str(agent.agent_type),
            "start": start,
            "end": (float(agent.tip_pos[0]), float(agent.tip_pos[1])),
            "angle": float(agent.tip_angle),
            "action": raw,
            "action_applied": float(np.clip(raw, lo, hi)),
            "reward": float(reward),
            "terms": [terms.get(k, 0.0) for k in TERM_NAMES],
            "seg_added": len(agent.segments) > n_seg,
            "agent_done": info.get("stop_reason") is not None,
            "stop_reason": str(info.get("stop_reason") or ""),
            "queue": ",".join(a.agent_type for a in u._agent_queue),
            "light_xy": light_xy,
            "gravity": gravity,
            "peaks": peaks,
            "obs": obs_before,
            "auxin": auxin,
            "cyto": cyto,
        })
        self._last_obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def episode(self) -> dict[str, Any]:
        """Return the recorded episode as a dict of numpy arrays (+ ``meta``)."""
        rows = self._rows
        if not rows:
            raise RuntimeError("No steps recorded — call reset() and step() first.")

        def arr(key: str, dtype=None) -> np.ndarray:
            return np.array([r[key] for r in rows], dtype=dtype)

        ep: dict[str, Any] = {
            "agent_id": arr("agent_id", np.int32),
            "agent_type": arr("agent_type"),
            "start": arr("start", np.float32),
            "end": arr("end", np.float32),
            "angle": arr("angle", np.float32),
            "action": arr("action", np.float32),
            "action_applied": arr("action_applied", np.float32),
            "reward": arr("reward", np.float32),
            "terms": arr("terms", np.float32),
            "term_names": np.array(TERM_NAMES),
            "seg_added": arr("seg_added", bool),
            "agent_done": arr("agent_done", bool),
            "stop_reason": arr("stop_reason"),
            "queue": arr("queue"),
            "light_xy": arr("light_xy", np.float32),
            "gravity": arr("gravity", np.float32),
            "peaks": arr("peaks", np.float32),
            "obs": arr("obs", np.float32),
            "meta": np.array(json.dumps(self._meta)),
        }
        if self.record_fields and rows[0]["auxin"] is not None:
            ep["auxin"] = np.stack([r["auxin"] for r in rows])
            ep["cyto"] = np.stack([r["cyto"] for r in rows])
        return ep

    def save(self, path: str | Path) -> Path:
        """Write the episode to ``path`` (.npz, compressed). Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.episode())
        return path


def load_episode(path: str | Path) -> dict[str, Any]:
    """Load an ``.npz`` written by :meth:`TrajectoryRecorder.save`."""
    with np.load(path, allow_pickle=False) as data:
        ep: dict[str, Any] = {k: data[k] for k in data.files}
    ep["meta"] = json.loads(str(ep["meta"]))
    return ep
