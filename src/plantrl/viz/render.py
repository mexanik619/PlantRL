"""Matplotlib replay of a recorded PlantEnv episode (see recorder.py).

Left:   the world — moisture / light / hormone layers, obstacles, plant segments
        coloured by agent type, the active tip + heading, handoff and spawn markers.
Right:  step info, cumulative reward split by term (this is the panel that shows
        *what the reward actually pays for*), and the 28-float observation the
        policy saw at this step, grouped by slice.

The renderer reads only the ``.npz`` episode — it never touches the env.
Set the backend *before* importing this module on headless machines:
    import matplotlib; matplotlib.use("Agg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .recorder import OBS_NAMES, OBS_SLICES

AGENT_COLORS = {
    "primary_root": "#7b4a2b",
    "lateral_root": "#c28a54",
    "primary_shoot": "#2e8b3a",
    "branch": "#8fd16a",
}
STOP_COLORS = {"WATER_FOUND": "#1f77b4", "LIGHT_FOUND": "#e6b800", "BUDGET": "#777777", "SUPPRESSED": "#d62728"}
TERM_COLORS = {"photo": "#e6b800", "hydro": "#1f77b4", "gravi": "#9467bd", "progress": "#17becf", "cost": "#999999", "goal": "#2ca02c"}
SLICE_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
ALL_FIELDS = ("moisture", "light", "auxin", "cyto")


# ----------------------------------------------------------------------
# Field helpers (mirror world/*.py; kept tiny so the renderer stays env-free)
# ----------------------------------------------------------------------

def _moisture_grid(X, Y, sources, peaks):
    total = np.zeros_like(X)
    for s, p in zip(sources, peaks):
        total += p * np.exp(-((X - s["cx"]) ** 2 + (Y - s["cy"]) ** 2) / (2.0 * s["sigma"] ** 2))
    return np.where(Y < 0.0, np.clip(total, 0.0, 1.0), np.nan)


def _light_grid(X, Y, src_xy, k, peak):
    d2 = (X - src_xy[0]) ** 2 + (Y - src_xy[1]) ** 2
    return np.where(Y >= 0.0, np.clip(peak / (1.0 + k * d2), 0.0, 1.0), np.nan)


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

def build_episode_view(ep, fig, fields=("moisture", "light"), every=1):
    """Lay the episode out on ``fig`` (cleared first). Returns ``(draw, frames, T)``.

    ``draw(i)`` paints step ``i`` onto this same figure, so one figure can be reused
    for many episodes (e.g. one video that contains a whole demo session).
    """
    unknown = set(fields) - set(ALL_FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(unknown)}; choose from {ALL_FIELDS}")
    if ("auxin" in fields or "cyto" in fields) and "auxin" not in ep:
        raise ValueError("this episode was recorded with record_fields=False; no hormone grids stored")

    meta = ep["meta"]
    T = len(ep["reward"])
    x0, x1, y0, y1 = meta["bounds"]
    types = [str(t) for t in ep["agent_type"]]
    stops = [str(s) for s in ep["stop_reason"]]
    queues = [str(q) for q in ep["queue"]]
    cum_total = np.cumsum(ep["reward"])
    term_names = [str(n) for n in ep["term_names"]]
    cum_terms = np.cumsum(ep["terms"], axis=0)

    # precomputed geometry for the plant
    segs = np.stack([ep["start"], ep["end"]], axis=1)              # [T, 2, 2]
    seg_colors = np.array([AGENT_COLORS.get(t, "#444444") for t in types])
    seg_widths = np.array([2.8 if t.startswith("primary") else 1.8 for t in types])
    first_step = {}
    for i, aid in enumerate(ep["agent_id"]):
        first_step.setdefault(int(aid), i)
    spawn_idx = np.array([i for aid, i in first_step.items() if aid >= 2], dtype=int)  # skip the two primaries

    fig.clf()
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[0.7, 1.0, 1.8],
                          hspace=0.45, wspace=0.14, left=0.045, right=0.985, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0])
    ax_info = fig.add_subplot(gs[0, 1]); ax_info.axis("off")
    ax_r = fig.add_subplot(gs[1, 1])
    ax_o = fig.add_subplot(gs[2, 1])

    # ---- world axes ----
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.axhspan(y0, 0.0, color="#d9c3a5", alpha=0.18, zorder=0)
    ax.axhline(0.0, color="#8b6b43", lw=1.2, zorder=1)
    ax.set_xlabel("x"); ax.set_ylabel("y   (soil at 0)")
    gx = np.linspace(x0, x1, 150); gy = np.linspace(y0, y1, 135)
    X, Y = np.meshgrid(gx, gy)
    ext = (x0, x1, y0, y1)
    layers: dict[str, Any] = {}
    if "moisture" in fields:
        layers["moisture"] = ax.imshow(np.full_like(X, np.nan), extent=ext, origin="lower", cmap="Blues",
                                       vmin=0, vmax=1, alpha=0.55, zorder=0.5, interpolation="bilinear")
    if "light" in fields:
        layers["light"] = ax.imshow(np.full_like(X, np.nan), extent=ext, origin="lower", cmap="YlOrBr",
                                    vmin=0, vmax=1, alpha=0.40, zorder=0.5, interpolation="bilinear")
    if "auxin" in fields:
        layers["auxin"] = ax.imshow(np.zeros((2, 2)), extent=ext, origin="lower", cmap="Reds",
                                    vmin=0, vmax=1, alpha=0.40, zorder=0.6, interpolation="bilinear")
    if "cyto" in fields:
        layers["cyto"] = ax.imshow(np.zeros((2, 2)), extent=ext, origin="lower", cmap="Purples",
                                   vmin=0, vmax=1, alpha=0.40, zorder=0.6, interpolation="bilinear")
    for (ox, oy, ow, oh) in meta["obstacles"]:
        ax.add_patch(Rectangle((ox, oy), ow, oh, fc="#555555", ec="k", alpha=0.75, zorder=2))
    light_marker = ax.scatter([], [], marker="*", s=380, c="#ffd21f", edgecolor="k", zorder=4)
    lc = LineCollection([], zorder=3)
    ax.add_collection(lc)
    spawn_sc = ax.scatter([], [], marker="^", s=70, c="white", edgecolor="k", zorder=4)
    done_sc = ax.scatter([], [], marker="o", s=130, facecolors="none", linewidths=2.2, zorder=4)
    tip_sc = ax.scatter([], [], s=70, c="red", edgecolor="k", zorder=6)
    head_ln, = ax.plot([], [], "k-", lw=1.4, zorder=6)
    coll_sc = ax.scatter([], [], marker="X", s=140, c="red", edgecolor="k", zorder=7)
    grav_q = ax.quiver([x0 + 2.0], [y0 + 3.0], [0.0], [-2.5], angles="xy", scale_units="xy", scale=1,
                       color="k", width=0.006, zorder=5)
    ax.text(x0 + 2.0, y0 + 3.4, "gravity", ha="center", fontsize=7)
    handles = [Line2D([], [], color=c, lw=3, label=n) for n, c in AGENT_COLORS.items()]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=c, mew=2, label=f"end: {n.lower()}")
                for n, c in STOP_COLORS.items()]
    handles.append(Line2D([], [], marker="^", ls="", mfc="white", mec="k", label="child agent starts"))
    ax.legend(handles=handles, loc="upper right", fontsize=6.5, framealpha=0.85, ncol=2)
    title = ax.set_title("", fontsize=10)

    # ---- info text ----
    info_txt = ax_info.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.5,
                            transform=ax_info.transAxes)

    # ---- reward panel (static curves + moving cursor) ----
    steps = np.arange(T)
    start_i = 0
    for i in range(1, T + 1):                                    # shade each agent's active span
        if i == T or ep["agent_id"][i] != ep["agent_id"][start_i]:
            ax_r.axvspan(start_i - 0.5, i - 0.5, color=AGENT_COLORS.get(types[start_i], "#444"), alpha=0.10, lw=0)
            start_i = i
    for k, name in enumerate(term_names):
        ax_r.plot(steps, cum_terms[:, k], color=TERM_COLORS.get(name, "k"), lw=1.3, label=name)
    ax_r.plot(steps, cum_total, color="k", lw=2.0, label="total")
    ax_r.axhline(0, color="#bbbbbb", lw=0.6)
    ax_r.set_xlim(-0.5, T - 0.5)
    ax_r.set_title("cumulative reward by term  (shaded = active agent)", fontsize=8.5)
    ax_r.tick_params(labelsize=7)
    ax_r.legend(fontsize=6.5, ncol=7, loc="upper left", frameon=False, handlelength=1.2, columnspacing=0.8)
    cursor = ax_r.axvline(0, color="red", lw=1.2)

    # ---- observation panel ----
    n_obs = len(OBS_NAMES)
    bar_colors = []
    for j in range(n_obs):
        for si, (a, b, _) in enumerate(OBS_SLICES):
            if a <= j < b:
                bar_colors.append(SLICE_COLORS[si])
    bars = ax_o.barh(np.arange(n_obs), np.zeros(n_obs), color=bar_colors, height=0.72)
    ax_o.set_yticks(np.arange(n_obs)); ax_o.set_yticklabels(OBS_NAMES, fontsize=5.8)
    ax_o.invert_yaxis(); ax_o.set_xlim(-1.05, 1.05); ax_o.axvline(0, color="k", lw=0.6)
    ax_o.tick_params(axis="x", labelsize=7)
    ax_o.set_title("policy observation at this step", fontsize=8.5)
    ax_o.legend(
        handles=[Line2D([], [], color=SLICE_COLORS[si], lw=5, label=name) for si, (_, _, name) in enumerate(OBS_SLICES)],
        loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(OBS_SLICES), fontsize=6.5,
        frameon=False, handlelength=1.0, columnspacing=1.0,
    )

    # ---------------- frame drawing ----------------
    def draw(i: int):
        i = int(min(max(i, 0), T - 1))
        peaks = ep["peaks"][i]
        if "moisture" in layers:
            layers["moisture"].set_data(_moisture_grid(X, Y, meta["moisture"], peaks))
        if "light" in layers:
            layers["light"].set_data(_light_grid(X, Y, ep["light_xy"][i], meta["light"]["falloff_k"], meta["light"]["intensity"]))
        if "auxin" in layers:
            layers["auxin"].set_data(ep["auxin"][i].astype(np.float32))
        if "cyto" in layers:
            layers["cyto"].set_data(ep["cyto"][i].astype(np.float32))
        light_marker.set_offsets([ep["light_xy"][i]])

        idx = np.where(ep["seg_added"][: i + 1])[0]
        lc.set_segments(segs[idx]); lc.set_color(seg_colors[idx]); lc.set_linewidth(seg_widths[idx])

        sp = spawn_idx[spawn_idx <= i]
        spawn_sc.set_offsets(ep["start"][sp] if len(sp) else np.empty((0, 2)))
        dn = np.where(ep["agent_done"][: i + 1] & ep["seg_added"][: i + 1])[0]
        done_sc.set_offsets(ep["end"][dn] if len(dn) else np.empty((0, 2)))
        done_sc.set_edgecolor([STOP_COLORS.get(stops[j], "#777") for j in dn] if len(dn) else "none")

        tip = ep["end"][i]
        tip_sc.set_offsets([tip]); tip_sc.set_facecolor(AGENT_COLORS.get(types[i], "red"))
        a = float(ep["angle"][i])
        head_ln.set_data([tip[0], tip[0] + 1.1 * np.cos(a)], [tip[1], tip[1] + 1.1 * np.sin(a)])
        hit = not bool(ep["seg_added"][i])
        coll_sc.set_offsets([tip] if hit else np.empty((0, 2)))
        gvec = ep["gravity"][i]
        grav_q.set_UVC([gvec[0] * 2.5], [gvec[1] * 2.5])

        stage, seed = meta["stage"], meta.get("seed")
        title.set_text(f"PlantRL replay — stage {stage}, seed {seed} — step {i + 1}/{T}")

        aid = int(ep["agent_id"][i])
        tilt = np.degrees(np.arctan2(gvec[0], -gvec[1]))
        info_txt.set_text(
            f"agent #{aid}  {types[i]}\n"
            f"tip      ({tip[0]:6.2f}, {tip[1]:6.2f})  heading {np.degrees(a) % 360:5.1f}°\n"
            f"action   {ep['action'][i]:+.3f} rad" + ("  (clipped)" if abs(ep['action'][i] - ep['action_applied'][i]) > 1e-6 else "") + "\n"
            f"reward   {ep['reward'][i]:+.3f}   total {cum_total[i]:+.2f}\n"
            f"gravity tilt {tilt:+.0f}°   light ({ep['light_xy'][i][0]:.1f}, {ep['light_xy'][i][1]:.1f})\n"
            f"queue    [{queues[i] or 'empty'}]\n"
            + (f"STOP     {stops[i]}" + ("  (collision)" if hit else "") if ep["agent_done"][i] else "")
        )
        cursor.set_xdata([i, i])
        for rect, v in zip(bars, ep["obs"][i]):
            rect.set_width(float(v))
        return ()

    frames = list(range(0, T, max(1, every)))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    return draw, frames, T


def render_episode(
    ep: dict[str, Any],
    out: str | Path | None = None,
    *,
    fields: Sequence[str] = ("moisture", "light"),
    fps: int = 12,
    every: int = 1,
    dpi: int = 80,
    frame: int | None = None,
    show: bool = False,
):
    """Render an episode.

    Args:
        ep:     Dict from ``load_episode`` / ``TrajectoryRecorder.episode``.
        out:    ``.gif`` / ``.mp4`` (animation) or ``.png`` (single frame).
                ``None`` + ``show=True`` opens an interactive window.
        fields: Background layers: any of moisture, light, auxin, cyto.
        every:  Draw every N-th step (the last step is always included).
        frame:  For ``.png`` output: which step to draw (default: last).
    """
    fig = plt.figure(figsize=(13, 7.4), dpi=dpi)
    draw, frames, T = build_episode_view(ep, fig, fields=fields, every=every)
    out = Path(out) if out else None
    suffix = out.suffix.lower() if out else ""
    if suffix == ".png" or (out is None and not show):
        draw(T - 1 if frame is None else frame)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi * 1.5)
        return fig

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False, repeat=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        elif suffix == ".mp4":
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("ffmpeg not found — install it or write a .gif instead")
            writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        else:
            raise ValueError("out must end in .gif, .mp4 or .png")
        anim.save(out, writer=writer, dpi=dpi)
        plt.close(fig)
    if show:
        plt.show()
    return anim