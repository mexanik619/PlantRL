"""plantrl-render: videos of a whole training run, and demos of a trained model.

Examples (after ``pip install -e '.[viz]'``):

    # the whole training session, as a video (+ a PNG of the final dashboard)
    plantrl-render train runs/train --out runs/train/training.mp4 --snapshot runs/train/final.png

    # trained model vs do-nothing, on the same 5 worlds, stage 2
    plantrl-render demo --model checkpoints/plant_sac_final.zip --stage 2 \
        --seeds 0 1 2 3 4 --baseline --out runs/demo_stage2.mp4

    # trained model only, GIF (no ffmpeg needed)
    plantrl-render demo --model checkpoints/best/best_model.zip --seeds 0 1 --out demo.gif
"""

from __future__ import annotations

import argparse

import matplotlib


def _do_nothing(obs, env):
    import numpy as np

    return np.zeros(1, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Render PlantRL training runs and policy demos")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="video of a whole training run")
    t.add_argument("run_dir", help="folder written by plantrl-train (contains episodes.csv)")
    t.add_argument("--out", default=None, help="default: <run_dir>/training.mp4")
    t.add_argument("--frames", type=int, default=120, help="episodes sampled across training")
    t.add_argument("--fps", type=int, default=12)
    t.add_argument("--snapshot", default=None, help="also save the final dashboard as a PNG")

    d = sub.add_parser("demo", help="roll out a trained model on fresh worlds and render it")
    d.add_argument("--model", required=True, help="SB3 SAC .zip")
    d.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    d.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    d.add_argument("--baseline", action="store_true", help="also show a do-nothing policy per world")
    d.add_argument("--out", default="runs/demo.mp4")
    d.add_argument("--fps", type=int, default=12)
    d.add_argument("--every", type=int, default=2, help="draw every N-th step")
    a = p.parse_args()

    matplotlib.use("Agg")
    from plantrl.viz.run_render import render_demo, render_training_snapshot, render_training_video

    if a.cmd == "train":
        out = a.out or f"{a.run_dir}/training.mp4"
        path = render_training_video(a.run_dir, out, frames=a.frames, fps=a.fps)
        print(f"[render] training video -> {path}")
        if a.snapshot:
            print(f"[render] final dashboard -> {render_training_snapshot(a.run_dir, a.snapshot)}")
        return

    from stable_baselines3 import SAC

    model = SAC.load(a.model, device="cpu")

    def trained(obs, env):
        return model.predict(obs, deterministic=True)[0]

    policies = [("trained model", trained)]
    if a.baseline:
        policies.append(("do nothing", _do_nothing))
    path = render_demo(policies, a.out, stage=a.stage, seeds=a.seeds, fps=a.fps, every=a.every)
    print(f"[render] demo -> {path}")


if __name__ == "__main__":
    main()