"""MoistureField — Gaussian moisture sources with gradient ∇M.

Multiple named moisture pockets exist in the soil (y < soil_boundary).
Gradient is computed via finite differences for use in hydrotropism rewards.
In stage 3 the sources deplete over time, forcing roots to keep moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MoistureSource:
    """A single Gaussian moisture pocket."""

    cx: float          # centre x
    cy: float          # centre y  (should be < soil_boundary, i.e. underground)
    peak: float        # max moisture value [0, 1]
    sigma: float = 2.0  # spatial spread


class MoistureField:
    """2D moisture field composed of Gaussian sources.

    Args:
        sources:        List of MoistureSource objects.
        soil_boundary:  y coordinate that separates above/below ground.
                        Above ground moisture is always 0.
        depletion_rate: How much peak decreases per ``deplete()`` call (stage 3).
        fd_h:           Step size for finite-difference gradient approximation.
    """

    def __init__(
        self,
        sources: list[MoistureSource] | None = None,
        soil_boundary: float = 0.0,
        depletion_rate: float = 0.002,
        fd_h: float = 0.1,
    ) -> None:
        self.sources: list[MoistureSource] = sources or []
        self.soil_boundary = soil_boundary
        self.depletion_rate = depletion_rate
        self.fd_h = fd_h

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_level(self, x: float, y: float) -> float:
        """Return total moisture ∈ [0, 1] at (x, y).

        Above soil_boundary always returns 0.
        """
        if y >= self.soil_boundary:
            return 0.0
        total = 0.0
        for src in self.sources:
            dx = x - src.cx
            dy = y - src.cy
            d_sq = dx * dx + dy * dy
            total += src.peak * np.exp(-d_sq / (2.0 * src.sigma ** 2))
        return float(np.clip(total, 0.0, 1.0))

    def get_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Return ∇M at (x, y) via central finite differences.

        Returns (grad_x, grad_y). Gradient below soil_boundary only;
        above ground returns (0, 0).
        """
        if y >= self.soil_boundary:
            return (0.0, 0.0)
        h = self.fd_h
        gx = (self.get_level(x + h, y) - self.get_level(x - h, y)) / (2.0 * h)
        gy = (self.get_level(x, y + h) - self.get_level(x, y - h)) / (2.0 * h)
        return (float(gx), float(gy))

    # ------------------------------------------------------------------
    # Dynamics (stage 3)
    # ------------------------------------------------------------------

    def deplete(self, dt: float = 1.0) -> None:
        """Reduce moisture source peaks over time (stage 3 dynamic soil)."""
        for src in self.sources:
            src.peak = max(0.0, src.peak - self.depletion_rate * dt)

    # ------------------------------------------------------------------
    # Reset / randomise
    # ------------------------------------------------------------------

    def reset(
        self,
        rng: np.random.Generator | None = None,
        n_sources: int = 3,
        world_width: float = 20.0,
        max_depth: float = 12.0,
        depleting: bool = False,
        sigma_range: tuple[float, float] = (1.5, 3.5),
    ) -> None:
        """Generate new random moisture sources for a new episode.

        Args:
            rng:         NumPy RNG.
            n_sources:   Number of Gaussian pockets.
            world_width: Lateral span of the world.
            max_depth:   Maximum underground depth (positive value; y = -max_depth).
            depleting:   If True enables per-step depletion (stage 3).
            sigma_range: (low, high) of each pocket's Gaussian spread. Wider pockets keep
                         the gradient non-zero far from the centre (see PlantEnv stage 1).
        """
        rng = rng or np.random.default_rng()
        self.sources = []
        for _ in range(n_sources):
            cx = float(rng.uniform(-world_width / 2, world_width / 2))
            cy = float(rng.uniform(-max_depth, self.soil_boundary - 0.5))
            peak = float(rng.uniform(0.6, 1.0))
            sigma = float(rng.uniform(*sigma_range))
            self.sources.append(MoistureSource(cx=cx, cy=cy, peak=peak, sigma=sigma))
        if depleting:
            self.depletion_rate = 0.002
        else:
            self.depletion_rate = 0.0

    def __repr__(self) -> str:
        return f"MoistureField(n_sources={len(self.sources)}, soil_boundary={self.soil_boundary})"
