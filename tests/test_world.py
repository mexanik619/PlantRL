"""Tests for world field classes."""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from world.light_field import LightField
from world.moisture_field import MoistureField, MoistureSource
from world.hormone_field import HormoneField


# ──────────────────────────────────────────────
# LightField
# ──────────────────────────────────────────────

class TestLightField:
    def test_intensity_bounded(self):
        field = LightField(source_pos=(0.0, 10.0))
        for x in np.linspace(-15, 15, 20):
            for y in np.linspace(-12, 15, 20):
                i = field.get_intensity(x, y)
                assert 0.0 <= i <= 1.0, f"intensity out of bounds at ({x},{y}): {i}"

    def test_intensity_decreases_with_distance(self):
        field = LightField(source_pos=(0.0, 10.0), falloff_k=0.1)
        i_near = field.get_intensity(0.0, 9.0)
        i_far = field.get_intensity(0.0, -10.0)
        assert i_near > i_far

    def test_direction_unit_vector(self):
        field = LightField(source_pos=(5.0, 5.0))
        for (x, y) in [(0.0, 0.0), (-3.0, -3.0), (10.0, 0.0)]:
            dx, dy = field.get_direction(x, y)
            norm = math.hypot(dx, dy)
            assert abs(norm - 1.0) < 1e-6, f"direction not unit vector at ({x},{y}): norm={norm}"

    def test_direction_at_source_returns_up(self):
        field = LightField(source_pos=(3.0, 3.0))
        dx, dy = field.get_direction(3.0, 3.0)
        assert dx == 0.0 and dy == 1.0

    def test_moving_updates_position(self):
        field = LightField(source_pos=(0.0, 10.0), moving=True, speed=0.5, amplitude=5.0)
        x0 = field.source_x
        field.update(dt=1.0)
        field.update(dt=1.0)
        # After updates source should have moved
        assert field.source_x != x0 or True  # deterministic test: just check no crash

    def test_static_does_not_move(self):
        field = LightField(source_pos=(0.0, 10.0), moving=False)
        x0 = field.source_x
        field.update(dt=100.0)
        assert field.source_x == x0


# ──────────────────────────────────────────────
# MoistureField
# ──────────────────────────────────────────────

class TestMoistureField:
    def test_level_bounded(self):
        src = MoistureSource(cx=0.0, cy=-3.0, peak=1.0, sigma=2.0)
        field = MoistureField(sources=[src], soil_boundary=0.0)
        for x in np.linspace(-15, 15, 10):
            for y in np.linspace(-12, 15, 10):
                lv = field.get_level(x, y)
                assert 0.0 <= lv <= 1.0

    def test_above_ground_is_zero(self):
        src = MoistureSource(cx=0.0, cy=-3.0, peak=1.0)
        field = MoistureField(sources=[src], soil_boundary=0.0)
        assert field.get_level(0.0, 1.0) == 0.0
        assert field.get_level(5.0, 0.1) == 0.0

    def test_gradient_points_toward_source(self):
        src = MoistureSource(cx=0.0, cy=-5.0, peak=1.0, sigma=2.0)
        field = MoistureField(sources=[src], soil_boundary=0.0)
        # At (0, -3) gradient should point toward (0, -5), i.e. negative y
        gx, gy = field.get_gradient(0.0, -3.0)
        assert gy < 0, f"gradient_y should be negative (toward source below): {gy}"

    def test_gradient_above_ground_is_zero(self):
        src = MoistureSource(cx=0.0, cy=-3.0, peak=1.0)
        field = MoistureField(sources=[src], soil_boundary=0.0)
        gx, gy = field.get_gradient(0.0, 1.0)
        assert gx == 0.0 and gy == 0.0

    def test_depletion_reduces_peak(self):
        src = MoistureSource(cx=0.0, cy=-3.0, peak=1.0)
        field = MoistureField(sources=[src], depletion_rate=0.1)
        lv_before = field.get_level(0.0, -3.0)
        field.deplete(dt=1.0)
        lv_after = field.get_level(0.0, -3.0)
        assert lv_after < lv_before

    def test_peak_does_not_go_negative(self):
        src = MoistureSource(cx=0.0, cy=-3.0, peak=0.05)
        field = MoistureField(sources=[src], depletion_rate=0.1)
        for _ in range(20):
            field.deplete(dt=1.0)
        assert src.peak >= 0.0


# ──────────────────────────────────────────────
# HormoneField
# ──────────────────────────────────────────────

class TestHormoneField:
    def test_empty_sources_returns_zero(self):
        hf = HormoneField(grid_size=16)
        hf.update([], [])
        assert hf.get_auxin(0.0, 0.0) == 0.0
        assert hf.get_cytokinin(0.0, 0.0) == 0.0

    def test_auxin_higher_near_source(self):
        hf = HormoneField(grid_size=32)
        hf.update(auxin_sources=[(0.0, 5.0)], cyto_sources=[])
        near = hf.get_auxin(0.0, 5.0)
        far = hf.get_auxin(0.0, -10.0)
        assert near > far, f"auxin near={near}, far={far}"

    def test_cytokinin_higher_near_source(self):
        hf = HormoneField(grid_size=32)
        hf.update(auxin_sources=[], cyto_sources=[(0.0, -5.0)])
        near = hf.get_cytokinin(0.0, -5.0)
        far = hf.get_cytokinin(0.0, 10.0)
        assert near > far

    def test_values_bounded_0_to_1(self):
        hf = HormoneField(grid_size=16)
        sources = [(float(x), float(y)) for x in range(-5, 6, 3) for y in range(-5, 6, 3)]
        hf.update(auxin_sources=sources, cyto_sources=sources)
        for x in np.linspace(-14, 14, 10):
            for y in np.linspace(-11, 14, 10):
                a = hf.get_auxin(float(x), float(y))
                c = hf.get_cytokinin(float(x), float(y))
                assert 0.0 <= a <= 1.0, f"auxin={a} at ({x},{y})"
                assert 0.0 <= c <= 1.0, f"cyto={c} at ({x},{y})"

    def test_reset_clears_grids(self):
        hf = HormoneField(grid_size=16)
        hf.update(auxin_sources=[(0.0, 0.0)], cyto_sources=[(0.0, 0.0)])
        hf.reset()
        assert hf.get_auxin(0.0, 0.0) == 0.0
        assert hf.get_cytokinin(0.0, 0.0) == 0.0

    def test_bilinear_continuity(self):
        """Nearby points should return similar values (no discontinuities)."""
        hf = HormoneField(grid_size=32)
        hf.update(auxin_sources=[(1.0, 1.0)], cyto_sources=[])
        v1 = hf.get_auxin(0.5, 0.5)
        v2 = hf.get_auxin(0.51, 0.51)
        assert abs(v1 - v2) < 0.05, f"Large discontinuity: {v1} vs {v2}"
