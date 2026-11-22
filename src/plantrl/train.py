"""train.py — SAC training with 3-stage curriculum for PlantRL.

Usage (after ``pip install -e .``):
    plantrl-train --stage 1 --total-steps 1000000
    python -m plantrl.train --stage 1 --total-steps 1000000

The curriculum automatically advances the stage when mean episode reward
exceeds the configured threshold for 10k consecutive steps.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from plantrl.environment.env import PlantEnv
from plantrl.viz.run_recorder import RunLogCallback, RunRecorder


# ──────────────────────────────────────────────
# Curriculum config
# ──────────────────────────────────────────────

CURRICULUM = {
    1: {
        "n_envs": 4,
        "advance_threshold": 0.5,   # mean ep reward to advance
        "eval_freq": 10_000,
    },
    2: {
        "n_envs": 4,
        "advance_threshold": 1.2,
        "eval_freq": 10_000,
    },
    3: {
        "n_envs": 4,      # keep same env count for all stages
        "advance_threshold": float("inf"),  # no advance from stage 3
        "eval_freq": 10_000,
    },
}


# ──────────────────────────────────────────────
# Curriculum advancement callback
# ──────────────────────────────────────────────

def set_env_stage(vec_env, stage: int) -> None:
    """Switch every worker env to ``stage`` (applies on each env's next reset).

    Uses ``env_method`` -> ``PlantEnv.set_stage``. ``vec_env.set_attr("stage", ...)``
    looks like it should work but only sets the attribute on the outer Monitor
    wrapper, so the real env silently stays on the old stage.
    """
    vec_env.env_method("set_stage", stage)


class CurriculumCallback(BaseCallback):
    """Advances curriculum stage when mean reward threshold is met.

    Monitors mean episode reward over the last ``window`` episodes.
    When it exceeds ``threshold`` every worker env is switched to the next
    stage in place (see ``set_env_stage``); the vec env itself is not rebuilt.
    """

    def __init__(
        self,
        model_ref: list,          # mutable 1-element list so we can update from callback
        stage_ref: list[int],
        stage_cfg: dict,
        window: int = 10_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.model_ref = model_ref
        self.stage_ref = stage_ref
        self.stage_cfg = stage_cfg
        self.window = window
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        # Collect episode rewards from info dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                if len(self._ep_rewards) > self.window:
                    self._ep_rewards = self._ep_rewards[-self.window:]

        stage = self.stage_ref[0]
        if stage >= 3:
            return True

        cfg = self.stage_cfg[stage]
        if (
            len(self._ep_rewards) >= 100
            and np.mean(self._ep_rewards[-100:]) >= cfg["advance_threshold"]
        ):
            new_stage = stage + 1
            self.stage_ref[0] = new_stage
            if self.verbose:
                print(f"\n[Curriculum] Advancing to stage {new_stage}")
            self._ep_rewards.clear()
            # Update stage on existing envs in-place (takes effect on
            # next natural episode reset, avoids env-swap Monitor issues)
            set_env_stage(self.model_ref[0].env, new_stage)

        return True


# ──────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────

def train(
    start_stage: int = 1,
    total_steps: int = 100_000,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
    run_dir: str | None = "runs/train",
    detail_every: int = 100,
) -> None:
    """``run_dir``: record every training episode there (None disables recording)."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg = CURRICULUM[start_stage]
    n_envs = cfg["n_envs"]

    print(f"[PlantRL] Starting at stage {start_stage} with {n_envs} envs")

    # ── Build training envs ──
    record = run_dir is not None
    train_env = make_vec_env(
        lambda: PlantEnv(stage=start_stage),
        n_envs=n_envs,
        seed=seed,
        wrapper_class=RunRecorder if record else None,
        wrapper_kwargs={"run_dir": run_dir, "detail_every": detail_every} if record else None,
    )

    # ── Eval env (single, deterministic) ──
    eval_env = make_vec_env(
        lambda: PlantEnv(stage=start_stage),
        n_envs=1,
        seed=seed + 999,
    )

    # ── SAC model ──
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_entropy=-1.0,           # per spec
        ent_coef="auto",               # auto-tuned temperature α
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
    )

    # ── Callbacks ──
    stage_ref = [start_stage]
    model_ref = [model]

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="plant_sac",
        verbose=1,
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoint_dir, "best"),
        log_path=log_dir,
        eval_freq=max(cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    curriculum_cb = CurriculumCallback(
        model_ref=model_ref,
        stage_ref=stage_ref,
        stage_cfg=CURRICULUM,
        verbose=1,
    )
    callbacks = [checkpoint_cb, eval_cb, curriculum_cb]
    if record:
        callbacks.append(RunLogCallback(run_dir))
    # ── Train ──
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        tb_log_name="SAC_plant",
        reset_num_timesteps=True,
    )

    model.save(os.path.join(checkpoint_dir, "plant_sac_final"))
    print("[PlantRL] Training complete. Model saved.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PlantRL SAC agent")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs/train",help="where every training episode is recorded")
    parser.add_argument("--no-record", action="store_true", help="disable episode recording")
    parser.add_argument("--detail-every", type=int, default=100,help="every Nth episode per env also stores hormone grids")
    args = parser.parse_args()

    train(
        start_stage=args.stage,
        total_steps=args.total_steps,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        run_dir=None if args.no_record else args.run_dir,
        detail_every=args.detail_every,
    )


if __name__ == "__main__":
    main()
