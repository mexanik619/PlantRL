"""PrimaryShootAgent — main upward shoot with strong phototropism."""

from __future__ import annotations

import math

from .base_agent import BaseAgent
from .types import AgentHandoff, StopReason, WorldSnapshot

# Goal: light intensity at tip exceeds this threshold
LIGHT_GOAL_THRESHOLD = 0.85

# Maximum upward height (world units above soil boundary y=0)
MAX_HEIGHT = 12.0


class PrimaryShootAgent(BaseAgent):
    """Primary shoot — grows upward seeking light.

    Tropism weights: phototropism > gravitropism (upward); no hydrotropism.
    Terminates when light > 0.85, max height reached, or budget exhausted.
    """

    W_PHOTO = 1.5
    W_HYDRO = 0.0
    W_GRAVI = 0.0   # gravity term dots with (0,-1): a positive weight would penalise growing UP
    IS_ROOT = False
    BRANCH_DEPTH = 0
    MAX_STEPS = 50
    STEP_LENGTH = 0.3

    def __init__(
        self,
        start_pos: tuple[float, float] = (0.0, 0.0),
        handoff: AgentHandoff | None = None,
        max_steps: int | None = None,
    ) -> None:
        # Primary shoot starts pointing upward (+π/2 radians)
        super().__init__(
            agent_type="primary_shoot",
            branch_depth=0,
            parent_angle=math.pi / 2,
            start_pos=start_pos,
            handoff=handoff,
            max_steps=max_steps,
        )

    def _goal_reached(self, world: WorldSnapshot) -> bool:
        height_reached = self.tip_pos[1] >= MAX_HEIGHT
        light_ok = world.light_intensity >= LIGHT_GOAL_THRESHOLD
        return light_ok or height_reached

    def _get_stop_reason(self) -> StopReason:
        if self.tip_pos[1] >= MAX_HEIGHT:
            return StopReason.BUDGET
        return StopReason.LIGHT_FOUND
