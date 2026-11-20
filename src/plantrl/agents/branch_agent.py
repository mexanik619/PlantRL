"""BranchAgent — shoot branch subject to apical dominance via auxin suppression."""

from __future__ import annotations

import math

from .base_agent import BaseAgent
from .types import AgentHandoff, StopReason, WorldSnapshot

LIGHT_GOAL_THRESHOLD = 0.80
MAX_HEIGHT = 12.0

# Branch is suppressed if local auxin exceeds this
AUXIN_SUPPRESSION_THRESHOLD = 0.7


class BranchAgent(BaseAgent):
    """Lateral shoot branch — phototropic, suppressed by high auxin concentration.

    Implements apical dominance: if auxin_local > threshold, the branch stops
    immediately with ``stop_reason = SUPPRESSED``. This causes the main shoot to
    dominate. When the main shoot tip is removed (auxin drops), branches activate.
    """

    W_PHOTO = 1.3
    W_HYDRO = 0.0
    W_GRAVI = 0.0   # see primary_shoot_agent: positive weight would penalise growing up
    IS_ROOT = False
    BRANCH_DEPTH = 1
    MAX_STEPS = 40
    STEP_LENGTH = 0.25

    def __init__(
        self,
        start_pos: tuple[float, float] = (0.0, 1.0),
        parent_angle: float = math.pi / 4,   # default: diagonal-upward
        handoff: AgentHandoff | None = None,
        max_steps: int | None = None,
        branch_depth: int = 1,
    ) -> None:
        super().__init__(
            agent_type="branch",
            branch_depth=branch_depth,
            parent_angle=parent_angle,
            start_pos=start_pos,
            handoff=handoff,
            max_steps=max_steps,
        )

    def _goal_reached(self, world: WorldSnapshot) -> bool:
        height_ok = self.tip_pos[1] >= MAX_HEIGHT
        light_ok = world.light_intensity >= LIGHT_GOAL_THRESHOLD
        return light_ok or height_ok

    def _get_stop_reason(self) -> StopReason:
        if self.tip_pos[1] >= MAX_HEIGHT:
            return StopReason.BUDGET
        return StopReason.LIGHT_FOUND

    def _extra_terminal(self, world: WorldSnapshot) -> bool:
        """Apical dominance: terminate immediately in high-auxin zones."""
        return world.auxin_local > AUXIN_SUPPRESSION_THRESHOLD
