"""Shared data structures for the agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class StopReason(IntEnum):
    """Why an agent stopped growing. Used for one-hot handoff encoding."""
    WATER_FOUND = 0
    LIGHT_FOUND = 1
    BUDGET      = 2
    SUPPRESSED  = 3


@dataclass
class AgentHandoff:
    """Context passed from a stopping agent to the next agent in the queue.

    All values default to 'neutral' so newly-spawned agents with no parent
    (e.g. the initial root/shoot) start from a clean slate.
    """
    stop_reason: StopReason = StopReason.BUDGET

    def to_onehot(self) -> list[float]:
        """Return a 4-float one-hot encoding of the stop reason."""
        v = [0.0, 0.0, 0.0, 0.0]
        v[int(self.stop_reason)] = 1.0
        return v


@dataclass
class GrowthSegment:
    """A recorded segment of plant growth (for rendering and global state)."""
    start_pos: tuple[float, float]
    end_pos:   tuple[float, float]
    is_root:   bool
    depth:     int          # branch_depth


@dataclass
class WorldSnapshot:
    """Lightweight view of world fields, passed into agents to build observations."""
    light_dir_x:      float
    light_dir_y:      float
    light_intensity:  float
    moisture_grad_x:  float
    moisture_grad_y:  float
    moisture_level:   float
    gravity_x:        float
    gravity_y:        float
    auxin_local:      float
    cytokinin_local:  float
    # Privileged info used ONLY for reward shaping (BaseAgent._compute_reward).
    # It is deliberately NOT part of the 28-float observation.
    water_target:     tuple[float, float] | None = None   # nearest live moisture pocket centre
    light_target:     tuple[float, float] | None = None   # light source position


@dataclass
class GlobalPlantState:
    """Aggregated state of the whole plant (bridging root↔shoot coordination)."""
    shoot_light_deficit: float = 0.0   # avg light starvation across shoot tips
    root_water_found:    float = 0.0   # avg moisture across root tips
    total_biomass_norm:  float = 0.0
    shoot_height_norm:   float = 0.0
    root_depth_norm:     float = 0.0

    def to_vec(self) -> list[float]:
        return [
            self.shoot_light_deficit,
            self.root_water_found,
            self.total_biomass_norm,
            self.shoot_height_norm,
            self.root_depth_norm,
        ]
