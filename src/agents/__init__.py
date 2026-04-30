"""Agents package — sequential multi-agent plant tip system."""

from .types import AgentHandoff, GlobalPlantState, GrowthSegment, StopReason, WorldSnapshot
from .base_agent import BaseAgent, OBS_DIM
from .agent_registry import spawn, registered_types, REGISTRY
from .primary_root_agent import PrimaryRootAgent
from .primary_shoot_agent import PrimaryShootAgent
from .lateral_root_agent import LateralRootAgent
from .branch_agent import BranchAgent

__all__ = [
    "AgentHandoff",
    "GlobalPlantState",
    "GrowthSegment",
    "StopReason",
    "WorldSnapshot",
    "BaseAgent",
    "OBS_DIM",
    "spawn",
    "registered_types",
    "REGISTRY",
    "PrimaryRootAgent",
    "PrimaryShootAgent",
    "LateralRootAgent",
    "BranchAgent",
]
