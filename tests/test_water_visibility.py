"""Stage 1 must let the root SENSE the water from the start (otherwise hydrotropism is unlearnable)."""

from __future__ import annotations

import math

import numpy as np

from plantrl.environment.env import PlantEnv


def test_water_gradient_is_visible_just_below_the_surface_in_stage_1():
    # With pocket sigma ~1.5 the gradient was ~0 for most of the root's path (0.00000 at the start).
    grads = []
    for seed in range(30):
        env = PlantEnv(stage=1)
        env.reset(seed=seed)
        gx, gy = env.unwrapped.moisture.get_gradient(0.0, -0.3)   # first point below the soil line
        grads.append(math.hypot(gx, gy))
    assert np.median(grads) > 0.02, f"median |grad| {np.median(grads):.4f}"
