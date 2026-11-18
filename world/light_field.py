"""LightField — continuous 2D directional light with intensity falloff.

Source position is fixed for stages 1-2. In stage 3 it moves along a
sinusoidal path so the shoot must actively track it.
"""

from __future__ import annotations

import math
import numpy as np


class LightField:
    """Continuous 2D light source with intensity falloff and optional motion.

    Args:
        source_pos: (x, y) initial position of the light source.
        intensity:  Peak intensity [0, 1].
        falloff_k:  Distance falloff constant. Intensity = 1 / (1 + k * d²).
        moving:     If True, source oscillates horizontally (stage 3).
        speed:      Horizontal oscillation speed (world-units / time-step) when moving.
        amplitude:  Half-width of oscillation path.
    """

    def __init__(
        self,
        source_pos: tuple[float, float] = (0.0, 10.0),
        intensity: float = 1.0,
        falloff_k: float = 0.05,
        moving: bool = False,
        speed: float = 0.05,
        amplitude: float = 5.0,
    ) -> None:
        self.base_x, self.base_y = source_pos
        self.source_x = self.base_x
        self.source_y = self.base_y
        self.intensity = intensity
        self.falloff_k = falloff_k
        self.moving = moving
        self.speed = speed
        self.amplitude = amplitude
        self._phase: float = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_intensity(self, x: float, y: float) -> float:
        """Return light intensity ∈ [0, 1] at world position (x, y)."""
        dx = self.source_x - x
        dy = self.source_y - y
        d_sq = dx * dx + dy * dy
        raw = self.intensity / (1.0 + self.falloff_k * d_sq)
        return float(np.clip(raw, 0.0, 1.0))

    def get_direction(self, x: float, y: float) -> tuple[float, float]:
        """Return unit vector pointing from (x, y) toward the light source.

        Returns (0, 1) (straight up) when the agent is at the source position.
        """
        dx = self.source_x - x
        dy = self.source_y - y
        dist = math.hypot(dx, dy)
        if dist < 1e-8:
            return (0.0, 1.0)
        return (dx / dist, dy / dist)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def update(self, dt: float = 1.0) -> None:
        """Advance the light source (only has effect when ``moving=True``)."""
        if not self.moving:
            return
        self._phase += self.speed * dt
        self.source_x = self.base_x + self.amplitude * math.sin(self._phase)

    # ------------------------------------------------------------------
    # Reset / randomise
    # ------------------------------------------------------------------

    def reset(
        self,
        rng: np.random.Generator | None = None,
        world_width: float = 20.0,
        moving: bool | None = None,
    ) -> None:
        """Randomise source position for a new episode.

        Args:
            rng:         NumPy RNG (for reproducibility with seeded envs).
            world_width: Used to clamp x positions.
            moving:      Override moving flag (e.g. for curriculum stage 3).
        """
        rng = rng or np.random.default_rng()
        self.base_x = float(rng.uniform(-world_width / 2, world_width / 2))
        self.base_y = float(rng.uniform(8.0, 15.0))
        self.source_x = self.base_x
        self.source_y = self.base_y
        self._phase = float(rng.uniform(0, 2 * math.pi))
        if moving is not None:
            self.moving = moving

    def __repr__(self) -> str:
        return (
            f"LightField(source=({self.source_x:.2f}, {self.source_y:.2f}), "
            f"intensity={self.intensity}, moving={self.moving})"
        )
