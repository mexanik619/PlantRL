"""visualize.py — record and replay PlantEnv episodes.

Examples (after ``pip install -e .[viz]``; `plantrl-viz` == `python -m plantrl.visualize`):
    # Random policy, stage 1, animated gif
    plantrl-viz --stage 1 --policy random --seed 0 --out runs/viz/random_s1.gif

    # "Do nothing" baseline (action = 0 every step) — what does the reward pay for?
    plantrl-viz --stage 1 --policy zero --out runs/viz/zero_s1.gif

    # Trained checkpoint, deterministic, last frame only
    plantrl-viz --stage 2 --policy checkpoint \
        --model checkpoints/best/best_model.zip --out runs/viz/best_s2.png

    # Re-render a saved recording with hormone layers (no env / no model needed)
    plantrl-viz --replay runs/viz/random_s1.npz --fields moisture auxin cyto --out hormones.gif
"""

from __future__ import annotations

import argparse
import os


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record / replay a PlantRL episode")
    p.add_argument("--replay", help="render an existing .npz recording instead of running the env")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--policy", default="random", choices=["random", "zero", "checkpoint"])
    p.add_argument("--model", help="SB3 SAC .zip (required for --policy checkpoint)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/viz/episode.gif", help=".gif, .mp4 or .png")
    p.add_argument("--save-npz", help="where to store the raw recording (default: next to --out)")
    p.add_argument("--fields", nargs="+", default=["moisture", "light"],
                   choices=["moisture", "light", "auxin", "cyto"])
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--every", type=int, default=1, help="draw every N-th step")
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--frame", type=int, help="step index for .png output (default: last)")
    p.add_argument("--show", action="store_true", help="open an interactive window")
    p.add_argument("--no-fields", action="store_true", help="do not store hormone grids (smaller .npz)")
    return p.parse_args()


def _make_policy(kind: str, env, model_path: str | None):
    import numpy as np

    if kind == "zero":
        return lambda obs: np.zeros(1, dtype=np.float32)
    if kind == "random":
        return lambda obs: env.action_space.sample()
    if not model_path:
        raise SystemExit("--policy checkpoint needs --model path/to/model.zip")
    from stable_baselines3 import SAC

    model = SAC.load(model_path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def record(stage: int, policy: str, model: str | None, seed: int, record_fields: bool = True):
    """Run one episode and return the finished TrajectoryRecorder."""
    from plantrl.environment.env import PlantEnv
    from plantrl.viz.recorder import TrajectoryRecorder

    env = TrajectoryRecorder(PlantEnv(stage=stage), record_fields=record_fields)
    env.action_space.seed(seed)
    act = _make_policy(policy, env, model)
    obs, _ = env.reset(seed=seed)
    done, n = False, 0
    while not done and n < 2000:                 # hard guard against a runaway episode
        obs, _, term, trunc, _ = env.step(act(obs))
        done, n = term or trunc, n + 1
    return env


def main() -> None:
    a = _parse()
    if not a.show:
        import matplotlib
        matplotlib.use("Agg")                    # headless-safe; must precede pyplot import
    from plantrl.viz.recorder import load_episode
    from plantrl.viz.render import render_episode

    if a.replay:
        ep = load_episode(a.replay)
    else:
        rec = record(a.stage, a.policy, a.model, a.seed, record_fields=not a.no_fields)
        npz = a.save_npz or os.path.splitext(a.out)[0] + ".npz"
        rec.save(npz)
        print(f"[viz] recording saved -> {npz}")
        ep = rec.episode()
        import json
        ep["meta"] = json.loads(str(ep["meta"]))

    T = len(ep["reward"])
    print(f"[viz] {T} steps, return {float(ep['reward'].sum()):.2f}, agents {int(ep['agent_id'].max()) + 1}")
    render_episode(ep, a.out, fields=a.fields, fps=a.fps, every=a.every, dpi=a.dpi,
                   frame=a.frame, show=a.show)
    print(f"[viz] rendered -> {a.out}")


if __name__ == "__main__":
    main()
