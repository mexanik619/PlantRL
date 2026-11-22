"""Renderers for whole training runs and for demos of a trained policy.

``render_training_video`` reads a run folder written by ``RunRecorder`` /
``RunLogCallback`` (``episodes.csv`` + ``episodes/*.npz``) and produces a video of the
whole session: learning curve, reward terms, branching and success rate fill in over
timesteps, while the world panel shows a sampled episode from that point in training.

``render_demo`` rolls out policies on fresh worlds and writes one video: title card,
the full animated episode, and a results card, for each seed and policy. Pass a
do-nothing policy next to the trained one to show what the policy adds.

Output: ``.mp4`` (needs ffmpeg) or ``.gif`` (no extra tools). ``.png`` snapshots are
also supported by ``render_training_snapshot``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from plantrl.viz.recorder import TERM_NAMES, TrajectoryRecorder, load_episode
from plantrl.viz.render import AGENT_COLORS, build_episode_view

FIG_SIZE = (14, 8)
WATER_STOP, LIGHT_STOP = "WATER_FOUND", "LIGHT_FOUND"
Policy = Callable[[np.ndarray, object], np.ndarray]   # (obs, env) -> action


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _writer_for(out: str | Path, fps: int) -> animation.AbstractMovieWriter:
    suffix = Path(out).suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg not found: install it (sudo pacman -S ffmpeg) or write a .gif")
        return animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    raise ValueError("out must end in .mp4 or .gif")


def _rolling(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) == 0:
        return y
    w = max(1, min(w, len(y)))
    c = np.cumsum(np.insert(y.astype(float), 0, 0.0))
    out = np.empty(len(y))
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _card(fig, lines: Sequence[tuple[str, int]]) -> None:
    fig.clf()
    ax = fig.add_subplot()
    ax.axis("off")
    n = len(lines)
    for k, (text, size) in enumerate(lines):
        y = 0.5 + (n - 1) / 2 * 0.12 - k * 0.12
        ax.text(0.5, y, text, ha="center", va="center", fontsize=size, transform=ax.transAxes)


# ----------------------------------------------------------------------
# training session
# ----------------------------------------------------------------------

def load_run(run_dir: str | Path) -> dict[str, np.ndarray]:
    """Read ``episodes.csv`` into column arrays (strings kept as lists)."""
    run_dir = Path(run_dir)
    with open(run_dir / "episodes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no episodes in {run_dir / 'episodes.csv'}")
    cols: dict[str, np.ndarray] = {
        "timestep": np.array([int(r["timestep"]) for r in rows]),
        "episode": np.array([int(r["episode"]) for r in rows]),
        "stage": np.array([int(r["stage"]) for r in rows]),
        "steps": np.array([int(r["steps"]) for r in rows]),
        "agents": np.array([int(r["agents"]) for r in rows]),
        "return": np.array([float(r["return"]) for r in rows]),
        "trajectory_file": [r["trajectory_file"] for r in rows],
        "stops": [json.loads(r["stop_reasons"] or "{}") for r in rows],
    }
    for k in TERM_NAMES:
        cols[k] = np.array([float(r[k]) for r in rows])
    cols["found"] = np.array([
        float(WATER_STOP in s or LIGHT_STOP in s) for s in cols["stops"]
    ])
    cols["run_dir"] = str(run_dir)
    return cols


def _draw_world(ax, ep: dict, meta: dict) -> None:
    x0, x1, y0, y1 = meta["bounds"]
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.axhspan(y0, 0.0, color="#d9c3a5", alpha=0.20, zorder=0)
    ax.axhline(0.0, color="#8b6b43", lw=1.2, zorder=1)
    for s, peak in zip(meta["moisture"], ep["peaks"][-1]):
        ax.add_patch(plt.Circle((s["cx"], s["cy"]), 1.5 * s["sigma"], color="#1f77b4",
                                alpha=0.03 + 0.07 * float(peak), zorder=0.5))
    for ox, oy, ow, oh in meta["obstacles"]:
        ax.add_patch(Rectangle((ox, oy), ow, oh, fc="#555555", ec="k", alpha=0.75, zorder=2))
    lx, ly = ep["light_xy"][-1]
    ax.scatter([lx], [ly], marker="*", s=320, c="#ffd21f", edgecolor="k", zorder=4)
    seg = np.stack([ep["start"], ep["end"]], axis=1)[ep["seg_added"]]
    types = [str(t) for t in ep["agent_type"][ep["seg_added"]]]
    colors = [AGENT_COLORS.get(t, "#444444") for t in types]
    widths = [2.8 if t.startswith("primary") else 1.8 for t in types]
    if len(seg):
        ax.add_collection(LineCollection(seg, colors=colors, linewidths=widths, zorder=3))
    ax.set_xlabel("x"); ax.set_ylabel("y (soil at 0)")
    ax.tick_params(labelsize=7)


def _dashboard(fig, cols: dict, upto: int, ep: dict, meta: dict, n_total: int) -> None:
    """Paint the training-session dashboard for episodes 0..upto (inclusive)."""
    fig.clf()
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.55, wspace=0.18,
                          left=0.05, right=0.97, top=0.92, bottom=0.07)
    ax_w = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])
    ax_b = fig.add_subplot(gs[2, 1])

    sl = slice(0, upto + 1)
    ts = cols["timestep"][sl]
    w = max(5, n_total // 25)

    # world: the sampled episode, finished
    _draw_world(ax_w, ep, meta)
    ax_w.set_title(
        f"episode {int(cols['episode'][upto])} · stage {int(cols['stage'][upto])} · "
        f"{int(cols['timestep'][upto]):,} steps trained",
        fontsize=10,
    )

    # learning curve
    ax_r.scatter(ts, cols["return"][sl], s=6, c="#999999", alpha=0.5, label="episode")
    ax_r.plot(ts, _rolling(cols["return"][sl], w), color="k", lw=2, label=f"mean ({w})")
    stage_change = np.where(np.diff(cols["stage"][sl]) != 0)[0] + 1
    for k in stage_change:
        ax_r.axvline(ts[k], color="#c44e52", ls="--", lw=1)
    ax_r.set_title("episode return", fontsize=9)
    ax_r.legend(fontsize=7, loc="lower right", frameon=False)
    ax_r.tick_params(labelsize=7)

    # reward terms, smoothed
    colors = {"photo": "#e6b800", "hydro": "#1f77b4", "gravi": "#9467bd",
              "progress": "#17becf", "cost": "#999999", "goal": "#2ca02c"}
    for k in TERM_NAMES:
        ax_t.plot(ts, _rolling(cols[k][sl], w), color=colors.get(k, "k"), lw=1.4, label=k)
    ax_t.axhline(0, color="#bbbbbb", lw=0.6)
    ax_t.set_title("reward per term (per episode, smoothed)", fontsize=9)
    ax_t.legend(fontsize=6, ncol=6, loc="upper left", frameon=False)
    ax_t.tick_params(labelsize=7)

    # branching and success
    ax_b.plot(ts, _rolling(cols["agents"][sl] - 2.0, w), color="#2e8b3a", lw=1.6,
              label="branches per episode")
    ax_b.set_ylabel("branches", fontsize=7, color="#2e8b3a")
    ax_b.tick_params(labelsize=7)
    ax2 = ax_b.twinx()
    ax2.plot(ts, _rolling(cols["found"][sl], w), color="#1f77b4", lw=1.4, label="water or light found")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("found rate", fontsize=7, color="#1f77b4")
    ax2.tick_params(labelsize=7)
    ax_b.set_title("branching and success (smoothed)", fontsize=9)
    ax_b.set_xlabel("timestep", fontsize=7)


def render_training_video(
    run_dir: str | Path,
    out: str | Path,
    *,
    frames: int = 120,
    fps: int = 12,
    hold_seconds: float = 2.0,
    dpi: int = 80,
) -> Path:
    """Video of the whole run. ``frames`` episodes are sampled evenly across training."""
    cols = load_run(run_dir)
    n = len(cols["episode"])
    picks = np.unique(np.linspace(0, n - 1, min(frames, n)).round().astype(int))
    run_dir = Path(run_dir)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)

    writer = _writer_for(out, fps)
    fig = plt.figure(figsize=FIG_SIZE, dpi=dpi)
    hold = int(hold_seconds * fps)
    with writer.saving(fig, str(out), dpi):
        for j, upto in enumerate(picks):
            ep = load_episode(run_dir / cols["trajectory_file"][upto])
            meta = ep["meta"]
            _dashboard(fig, cols, int(upto), ep, meta, n)
            writer.grab_frame()
            if j == len(picks) - 1:
                for _ in range(hold):
                    writer.grab_frame()
    plt.close(fig)
    return out


def render_training_snapshot(run_dir: str | Path, out: str | Path, dpi: int = 110) -> Path:
    """Single PNG of the dashboard at the end of training."""
    cols = load_run(run_dir)
    last = len(cols["episode"]) - 1
    ep = load_episode(Path(run_dir) / cols["trajectory_file"][last])
    fig = plt.figure(figsize=FIG_SIZE, dpi=dpi)
    _dashboard(fig, cols, last, ep, ep["meta"], len(cols["episode"]))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# demo of trained policies
# ----------------------------------------------------------------------

def rollout(policy: Policy, stage: int, seed: int, max_steps: int = 2000) -> dict:
    """One episode on a fresh world. Returns the recorded episode dict (+ meta)."""
    from plantrl.environment.env import PlantEnv

    env = TrajectoryRecorder(PlantEnv(stage=stage), record_fields=True)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    done, n = False, 0
    while not done and n < max_steps:
        obs, _, term, trunc, _ = env.step(policy(obs, env))
        done, n = term or trunc, n + 1
    ep = env.episode()
    ep["meta"] = json.loads(str(ep["meta"]))
    return ep


def render_demo(
    policies: Sequence[tuple[str, Policy]],
    out: str | Path,
    *,
    stage: int = 1,
    seeds: Sequence[int] = (0, 1, 2),
    fps: int = 12,
    every: int = 2,
    card_seconds: float = 2.0,
    dpi: int = 80,
) -> Path:
    """Video: for each seed, each policy on the same world, with title and result cards."""
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    writer = _writer_for(out, fps)
    fig = plt.figure(figsize=FIG_SIZE, dpi=dpi)
    hold = max(1, int(card_seconds * fps))
    names = " vs ".join(name for name, _ in policies)

    with writer.saving(fig, str(out), dpi):
        _card(fig, [("PlantRL", 34), (f"stage {stage} · {names}", 18),
                    ("same worlds for every policy", 14)])
        for _ in range(hold * 2):
            writer.grab_frame()
        for seed in seeds:
            for name, policy in policies:
                ep = rollout(policy, stage, seed)
                stops = {str(s) for s in ep["stop_reason"] if s}
                _card(fig, [(name, 30), (f"world seed {seed}", 16)])
                for _ in range(hold):
                    writer.grab_frame()
                view_draw, frames, T = build_episode_view(ep, fig, every=every)
                for i in frames:
                    view_draw(i)
                    writer.grab_frame()
                agents = int(ep["agent_id"].max()) + 1
                _card(fig, [
                    (name, 26),
                    (f"return {float(ep['reward'].sum()):+.1f}   ·   {T} steps", 18),
                    (f"water found: {'yes' if WATER_STOP in stops else 'no'}   ·   "
                     f"light found: {'yes' if LIGHT_STOP in stops else 'no'}", 18),
                    (f"branches: {max(agents - 2, 0)}", 16),
                ])
                for _ in range(hold):
                    writer.grab_frame()
    plt.close(fig)
    return out