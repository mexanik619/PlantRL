"""HormoneField — auxin + cytokinin diffusion precomputed on 32×32 coarse grid.

Both hormones are modelled as exponential decay from their source tips:

    concentration(x, y) = Σ  rate · exp(−decay · distance)

Precomputing on a coarse grid and using bilinear interpolation keeps the per-step
cost constant regardless of the number of sources, enabling CPU-friendly training.

Auxin   → produced at shoot tips, flows downward, decays fast.
Cytokinin → produced at root  tips, flows upward,  decays slow.
"""

from __future__ import annotations

import numpy as np


class HormoneField:
    """Precomputed auxin + cytokinin diffusion fields.

    Args:
        grid_size:      Resolution of the coarse grid (default 32×32).
        world_bounds:   (x_min, x_max, y_min, y_max) of the world.
        auxin_rate:     Emission rate per auxin source.
        auxin_decay:    Spatial decay constant for auxin (faster decay = shorter range).
        cyto_rate:      Emission rate per cytokinin source.
        cyto_decay:     Spatial decay constant for cytokinin (slower = longer range).
    """

    def __init__(
        self,
        grid_size: int = 32,
        world_bounds: tuple[float, float, float, float] = (-15.0, 15.0, -12.0, 15.0),
        auxin_rate: float = 1.0,
        auxin_decay: float = 0.4,
        cyto_rate: float = 1.0,
        cyto_decay: float = 0.2,
    ) -> None:
        self.grid_size = grid_size
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        self.auxin_rate = auxin_rate
        self.auxin_decay = auxin_decay
        self.cyto_rate = cyto_rate
        self.cyto_decay = cyto_decay

        # Precompute grid cell centre coordinates (shape: [G, G])
        xs = np.linspace(self.x_min, self.x_max, grid_size)
        ys = np.linspace(self.y_min, self.y_max, grid_size)
        self._grid_x, self._grid_y = np.meshgrid(xs, ys)  # [G, G]

        # Initialise flat grids
        self._auxin_grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._cyto_grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.float32)

    # ------------------------------------------------------------------
    # Grid update
    # ------------------------------------------------------------------

    def update(
        self,
        auxin_sources: list[tuple[float, float]],
        cyto_sources: list[tuple[float, float]],
    ) -> None:
        """Recompute both hormone grids from current tip positions.

        Args:
            auxin_sources:  List of (x, y) shoot-tip positions.
            cyto_sources:   List of (x, y) root-tip positions.
        """
        self._auxin_grid = self._compute_grid(
            auxin_sources, self.auxin_rate, self.auxin_decay
        )
        self._cyto_grid = self._compute_grid(
            cyto_sources, self.cyto_rate, self.cyto_decay
        )

    def _compute_grid(
        self,
        sources: list[tuple[float, float]],
        rate: float,
        decay: float,
    ) -> np.ndarray:
        """Vectorised hormone grid computation.

        Returns an ndarray of shape [grid_size, grid_size] with values ∈ [0, 1].
        """
        if not sources:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Broadcast: [G, G, 1] - [1, 1, N] → [G, G, N]
        srcs = np.array(sources, dtype=np.float32)          # [N, 2]
        sx, sy = srcs[:, 0], srcs[:, 1]                     # [N]

        dx = self._grid_x[:, :, np.newaxis] - sx[np.newaxis, np.newaxis, :]  # [G,G,N]
        dy = self._grid_y[:, :, np.newaxis] - sy[np.newaxis, np.newaxis, :]

        dist = np.sqrt(dx ** 2 + dy ** 2)                   # [G, G, N]
        contrib = rate * np.exp(-decay * dist)               # [G, G, N]
        total = contrib.sum(axis=-1)                         # [G, G]
        return np.clip(total, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Sampling (bilinear interpolation)
    # ------------------------------------------------------------------

    def get_auxin(self, x: float, y: float) -> float:
        """Return auxin concentration at (x, y) via bilinear interpolation."""
        return float(self._bilinear(self._auxin_grid, x, y))

    def get_cytokinin(self, x: float, y: float) -> float:
        """Return cytokinin concentration at (x, y) via bilinear interpolation."""
        return float(self._bilinear(self._cyto_grid, x, y))

    def _bilinear(self, grid: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation on the coarse grid.

        Returns the value clamped to [0, 1]. Out-of-bounds queries return 0.
        """
        G = self.grid_size

        # Map world coords → fractional grid indices
        tx = (x - self.x_min) / (self.x_max - self.x_min) * (G - 1)
        ty = (y - self.y_min) / (self.y_max - self.y_min) * (G - 1)

        # Clamp to grid
        tx = float(np.clip(tx, 0.0, G - 1.0001))
        ty = float(np.clip(ty, 0.0, G - 1.0001))

        x0, y0 = int(tx), int(ty)
        x1, y1 = x0 + 1, y0 + 1
        fx, fy = tx - x0, ty - y0

        # Grid is indexed [row=y, col=x]
        v00 = grid[y0, x0]
        v10 = grid[y0, x1]
        v01 = grid[y1, x0]
        v11 = grid[y1, x1]

        val = (
            v00 * (1 - fx) * (1 - fy)
            + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy
            + v11 * fx * fy
        )
        return float(np.clip(val, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear both hormone grids (call at episode start)."""
        self._auxin_grid[:] = 0.0
        self._cyto_grid[:] = 0.0

    def __repr__(self) -> str:
        return (
            f"HormoneField(grid={self.grid_size}×{self.grid_size}, "
            f"auxin_decay={self.auxin_decay}, cyto_decay={self.cyto_decay})"
        )
