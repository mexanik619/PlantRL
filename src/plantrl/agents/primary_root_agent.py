"""PrimaryRootAgent — main downward root with strong gravitropism and hydrotropism."""

from __future__ import annotations

import math

from .base_agent import BaseAgent
from .types import AgentHandoff, StopReason, WorldSnapshot

# Goal: moisture level at tip exceeds this threshold
MOISTURE_GOAL_THRESHOLD = 0.8

# Maximum downward depth (world units below soil boundary y=0)
MAX_DEPTH = 10.0


class PrimaryRootAgent(BaseAgent):
    """Primary root — grows downward seeking water.

    Tropism weights: hydrotropism > gravitropism; no phototropism.
    Terminates when moisture > 0.8, max depth reached, or budget exhausted.
    """

    W_PHOTO = 0.0
    W_HYDRO = 1.5
    W_GRAVI = 0.1
    IS_ROOT = True
    BRANCH_DEPTH = 0
    MAX_STEPS = 50
    STEP_LENGTH = 0.3

    def __init__(
        self,
        start_pos: tuple[float, float] = (0.0, 0.0),
        handoff: AgentHandoff | None = None,
        max_steps: int | None = None,
    ) -> None:
        # Primary root starts pointing downward (−π/2 radians)
        super().__init__(
            agent_type="primary_root",
            branch_depth=0,
            parent_angle=-math.pi / 2,
            start_pos=start_pos,
            handoff=handoff,
            max_steps=max_steps,
        )

    def _goal_reached(self, world: WorldSnapshot) -> bool:
        depth_reached = self.tip_pos[1] <= -MAX_DEPTH
        moisture_ok = world.moisture_level >= MOISTURE_GOAL_THRESHOLD
        return moisture_ok or depth_reached

    def _get_stop_reason(self) -> StopReason:
        # Distinguish between goal (water) and depth limit
        if self.tip_pos[1] <= -MAX_DEPTH:
            return StopReason.BUDGET
        return StopReason.WATER_FOUND
