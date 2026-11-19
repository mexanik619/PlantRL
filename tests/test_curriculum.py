"""Regression tests: the curriculum must really change the stage inside the worker envs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from plantrl.environment.env import PlantEnv
from plantrl.train import CURRICULUM, CurriculumCallback, set_env_stage


def _n_obstacles(venv) -> list[int]:
    # `_obstacles` lives on PlantEnv (not on the Monitor wrapper), so this reads the real env.
    return [len(o) for o in venv.env_method("get_wrapper_attr", "_obstacles")]


@pytest.fixture()
def venv():
    v = make_vec_env(lambda: PlantEnv(stage=1), n_envs=2, vec_env_cls=DummyVecEnv, seed=0)
    v.reset()
    yield v
    v.close()


def test_set_env_stage_reaches_the_real_env(venv):
    assert _n_obstacles(venv) == [0, 0]            # stage 1: no obstacles
    set_env_stage(venv, 3)
    venv.reset()                                    # stage change applies on next reset
    assert min(_n_obstacles(venv)) >= 4             # stage 3: 4-8 obstacles


def test_set_stage_validates(venv):
    with pytest.raises(Exception):
        venv.env_method("set_stage", 4)


def test_callback_advances_stage_in_the_envs(venv):
    cb = CurriculumCallback(
        model_ref=[SimpleNamespace(env=venv)], stage_ref=[2], stage_cfg=CURRICULUM, verbose=0
    )
    cb.locals = {"infos": [{"episode": {"r": 1e6}}]}   # far above any advance threshold
    for _ in range(100):                                # needs >= 100 finished episodes
        cb._on_step()
    assert cb.stage_ref[0] == 3
    venv.reset()
    assert min(_n_obstacles(venv)) >= 4
