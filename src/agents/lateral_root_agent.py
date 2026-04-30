"""LateralRootAgent — secondary root with shorter budget and strong hydrotropism."""

from __future__ import annotations

import math

from .base_agent import BaseAgent
from .types import AgentHandoff, StopReason, WorldSnapshot

MOISTURE_GOAL_THRESHOLD = 0.7   # slightly lower bar than primary root
MAX_DEPTH = 10.0


class LateralRootAgent(BaseAgent):
    """Lateral root — branches off from primary root, shorter step budget.

    Starts at a (typically) horizontal angle biased by parent orientation.
    Hydrotropism dominates; gravitropism still present but weaker.
    """

    W_PHOTO = 0.0
    W_HYDRO = 1.8   # even stronger hydrotropism than primary root
    W_GRAVI = 0.4
    IS_ROOT = True
    BRANCH_DEPTH = 1
    MAX_STEPS = 30   # shorter budget per spec
    STEP_LENGTH = 0.25

    def __init__(
        self,
        start_pos: tuple[float, float] = (0.0, -1.0),
        parent_angle: float = -math.pi / 4,   # default: diagonal-downward
        handoff: AgentHandoff | None = None,
        max_steps: int | None = None,
        branch_depth: int = 1,
    ) -> None:
        super().__init__(
            agent_type="lateral_root",
            branch_depth=branch_depth,
            parent_angle=parent_angle,
            start_pos=start_pos,
            handoff=handoff,
            max_steps=max_steps,
        )

    def _goal_reached(self, world: WorldSnapshot) -> bool:
        depth_ok = self.tip_pos[1] <= -MAX_DEPTH
        moisture_ok = world.moisture_level >= MOISTURE_GOAL_THRESHOLD
        return moisture_ok or depth_ok

    def _get_stop_reason(self) -> StopReason:
        if self.tip_pos[1] <= -MAX_DEPTH:
            return StopReason.BUDGET
        return StopReason.WATER_FOUND
