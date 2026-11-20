"""BaseAgent — shared logic for all plant meristem/tip agents.

Subclasses override:
  - ``reward_weights`` property → (w_photo, w_hydro, w_gravi)
  - ``is_root`` property
  - ``max_steps`` property
  - ``_goal_reached(world)`` → bool
  - ``_get_stop_reason()`` → StopReason  (called when _goal_reached is True)
  - ``_extra_terminal(world)`` → bool  (optional suppression / depth checks)

The 28-float observation vector layout is fixed by the spec:
  [0:8]   tropism signals
  [8:14]  geometry
  [14:16] hormonal
  [16:19] identity + temporal
  [19:23] handoff (one-hot stop reason)
  [23:28] global plant state
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .types import (
    AgentHandoff,
    GlobalPlantState,
    GrowthSegment,
    StopReason,
    WorldSnapshot,
)

if TYPE_CHECKING:
    pass

# Observation space dimensionality (must match spec)
OBS_DIM = 28

# Action bounds (radians) (Angle of bending in radians)
ACTION_MIN = -0.35
ACTION_MAX = 0.35

# Normalisation constants
MAX_HEIGHT = 15.0   # world units above soil
MAX_DEPTH  = 12.0   # world units below soil


class BaseAgent:
    """Abstract base for all plant-tip agents.

    Args:
        agent_type:    String key used for logging and registry lookup.
        branch_depth:  0 = primary, 1 = lateral/branch, 2 = secondary...
        parent_angle:  Angle of the parent segment (radians). 0 = right.
        start_pos:     (x, y) world coordinates of the meristem tip.
        handoff:       AgentHandoff from the parent agent (or neutral default).
        max_steps:     Growth step budget before forced termination.
    """

    # ------------------------------------------------------------------
    # Overrideable class-level defaults (subclasses should override)
    # ------------------------------------------------------------------
    W_PHOTO: float = 0.0
    W_HYDRO: float = 0.0
    W_GRAVI: float = 0.1       # mild bias only: a big weight pays for growing straight and ignoring targets
    W_PROGRESS: float = 1.0    # reward per world unit moved closer to the target (water for roots, light for shoots)
    GOAL_BONUS: float = 10.0   # paid only for a real water / light find (see step())
    ENERGY_COST: float = 0.01
    COLLISION_PENALTY: float = 0.5
    STEP_LENGTH: float = 0.3   # world units per growth step
    IS_ROOT: bool = False
    BRANCH_DEPTH: int = 0
    MAX_STEPS: int = 50

    def __init__(
        self,
        agent_type: str,
        branch_depth: int | None = None,
        parent_angle: float = math.pi / 2,  # default: pointing straight up
        start_pos: tuple[float, float] = (0.0, 0.0),
        handoff: AgentHandoff | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.agent_type = agent_type
        self.branch_depth = branch_depth if branch_depth is not None else self.BRANCH_DEPTH
        self.parent_angle = parent_angle
        self.tip_pos: tuple[float, float] = start_pos
        self.handoff = handoff or AgentHandoff()
        self.max_steps = max_steps if max_steps is not None else self.MAX_STEPS

        # Mutable state
        self.tip_angle: float = parent_angle   # current heading (radians)
        self.steps_taken: int = 0
        self.done: bool = False
        self.stop_reason: StopReason = StopReason.BUDGET
        self.segments: list[GrowthSegment] = []
        # Per-term breakdown of the last step's shaped reward (read by viz/recorder)
        self.last_reward_terms: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Observation builder — 28 floats (FIXED LAYOUT per spec)
    # ------------------------------------------------------------------

    def build_obs(
        self,
        world: WorldSnapshot,
        global_state: GlobalPlantState,
        norm_x: float = 0.0,
        norm_y: float = 0.0,
    ) -> np.ndarray:
        """Assemble the 28-float observation vector.

        Args:
            world:       Pre-computed world snapshot at current tip position.
            global_state: Aggregated plant-level statistics.
            norm_x:      tip_x / world_half_width  ∈ [-1, 1].
            norm_y:      tip_y / normalisation_range ∈ [-1, 1].
        """
        obs: list[float] = []

        # [0:8] Tropism signals
        obs.append(world.light_dir_x)
        obs.append(world.light_dir_y)
        obs.append(world.light_intensity)
        obs.append(world.moisture_grad_x)
        obs.append(world.moisture_grad_y)
        obs.append(world.moisture_level)
        obs.append(world.gravity_x)
        obs.append(world.gravity_y)

        # [8:14] Geometry (sin/cos to avoid ±π discontinuity)
        obs.append(math.sin(self.tip_angle))
        obs.append(math.cos(self.tip_angle))
        obs.append(math.sin(self.parent_angle))
        obs.append(math.cos(self.parent_angle))
        obs.append(float(np.clip(norm_x, -1.0, 1.0)))
        obs.append(float(np.clip(norm_y, -1.0, 1.0)))

        # [14:16] Hormonal
        obs.append(world.auxin_local)
        obs.append(world.cytokinin_local)

        # [16:19] Identity + temporal
        obs.append(1.0 if self.IS_ROOT else 0.0)
        obs.append(min(self.branch_depth / 4.0, 1.0))
        budget_used = self.steps_taken / max(self.max_steps, 1)
        obs.append(float(np.clip(budget_used, 0.0, 1.0)))

        # [19:23] Handoff one-hot
        obs.extend(self.handoff.to_onehot())

        # [23:28] Global plant state
        obs.extend(global_state.to_vec())

        assert len(obs) == OBS_DIM, f"Obs dim mismatch: {len(obs)} != {OBS_DIM}"
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core growth step
    # ------------------------------------------------------------------

    def step(
        self,
        delta_theta: float,
        world: WorldSnapshot,
        obstacles: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[np.ndarray | None, float, bool, StopReason]:
        """Execute one growth step.

        Args:
            delta_theta: Action ∈ [ACTION_MIN, ACTION_MAX] in radians.
            world:       World snapshot at the CURRENT tip position.
            obstacles:   List of AABB rects (x, y, w, h). Checked post-move.

        Returns:
            (new_tip_pos_array_or_None, reward, done, stop_reason)
            new_tip_pos is None only when ``done`` is True due to collision.
        """
        # Clamp action
        delta_theta = float(np.clip(delta_theta, ACTION_MIN, ACTION_MAX))

        # Update heading
        self.tip_angle += delta_theta

        # Compute new position
        dx = self.STEP_LENGTH * math.cos(self.tip_angle)
        dy = self.STEP_LENGTH * math.sin(self.tip_angle)
        old_pos = self.tip_pos
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy

        # Collision check against AABB obstacles
        collision = False
        if obstacles:
            for (ox, oy, ow, oh) in obstacles:
                if ox <= new_x <= ox + ow and oy <= new_y <= oy + oh:
                    collision = True
                    break

        self.steps_taken += 1

        # --- Compute reward ---
        reward = self._compute_reward(dx, dy, world, collision)

        if collision:
            self.done = True
            self.stop_reason = StopReason.BUDGET
            return None, reward, True, self.stop_reason

        # Update position and record segment
        new_pos = (new_x, new_y)
        self.segments.append(
            GrowthSegment(
                start_pos=old_pos,
                end_pos=new_pos,
                is_root=self.IS_ROOT,
                depth=self.branch_depth,
            )
        )
        self.tip_pos = new_pos

        # --- Terminal checks ---
        if self._goal_reached(world):
            self.stop_reason = self._get_stop_reason()
            # Only a real find pays. Hitting the depth/height cap (stop reason BUDGET)
            # must not, otherwise growing straight collects the bonus for free.
            if self.stop_reason in (StopReason.WATER_FOUND, StopReason.LIGHT_FOUND):
                reward += self.GOAL_BONUS
            self.done = True
        elif self._extra_terminal(world):
            self.stop_reason = StopReason.SUPPRESSED
            self.done = True
        elif self.steps_taken >= self.max_steps:
            self.stop_reason = StopReason.BUDGET
            self.done = True

        return np.array(new_pos, dtype=np.float32), reward, self.done, self.stop_reason

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        dx: float,
        dy: float,
        world: WorldSnapshot,
        collision: bool,
    ) -> float:
        """Compute per-step reward from tropism alignment + energy + collision."""
        step_len = math.hypot(dx, dy) + 1e-8
        gx, gy = dx / step_len, dy / step_len  # normalised growth direction

        photo = hydro = progress = 0.0

        # Progress toward the goal: potential-based shaping, Phi(s) = -distance to target.
        # Uses the true target position (privileged, reward-only); tip_pos is still the
        # position BEFORE this step. A collision means the tip did not move.
        if self.W_PROGRESS > 0 and not collision:
            target = world.water_target if self.IS_ROOT else world.light_target
            if target is not None:
                x, y = self.tip_pos
                d_old = math.hypot(target[0] - x, target[1] - y)
                d_new = math.hypot(target[0] - (x + dx), target[1] - (y + dy))
                progress = self.W_PROGRESS * (d_old - d_new)

        # Phototropism (shoot only via W_PHOTO)
        if self.W_PHOTO > 0:
            photo = self.W_PHOTO * (
                gx * world.light_dir_x + gy * world.light_dir_y
            ) * world.light_intensity

        # Hydrotropism (root only via W_HYDRO)
        if self.W_HYDRO > 0:
            hydro = self.W_HYDRO * (
                gx * world.moisture_grad_x + gy * world.moisture_grad_y
            )

        # Gravitropism (all agents). Dotted with the world gravity vector, so a positive
        # weight rewards growing DOWN. Shoots/branches therefore set W_GRAVI = 0 instead
        # of a positive weight, which would penalise growing up.
        gravi = self.W_GRAVI * (gx * world.gravity_x + gy * world.gravity_y)

        # Running costs
        cost = -self.ENERGY_COST
        if collision:
            cost -= self.COLLISION_PENALTY

        self.last_reward_terms = {
            "photo": float(photo),
            "hydro": float(hydro),
            "gravi": float(gravi),
            "progress": float(progress),
            "cost": float(cost),
        }
        return float(photo + hydro + gravi + progress + cost)

    # ------------------------------------------------------------------
    # Terminal condition hooks (override in subclasses)
    # ------------------------------------------------------------------

    def _goal_reached(self, world: WorldSnapshot) -> bool:  # noqa: ARG002
        """Return True when the agent's primary goal is satisfied."""
        return False

    def _get_stop_reason(self) -> StopReason:
        return StopReason.BUDGET

    def _extra_terminal(self, world: WorldSnapshot) -> bool:  # noqa: ARG002
        """Return True for auxiliary termination (e.g. auxin suppression)."""
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_tip_pos(self) -> tuple[float, float]:
        return self.tip_pos

    def __repr__(self) -> str:
        x, y = self.tip_pos
        return (
            f"{self.__class__.__name__}("
            f"depth={self.branch_depth}, "
            f"pos=({x:.2f},{y:.2f}), "
            f"steps={self.steps_taken}/{self.max_steps})"
        )
