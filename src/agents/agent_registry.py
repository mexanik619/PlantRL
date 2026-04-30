"""agent_registry.py — maps agent_type string → agent class.

Adding a new agent type requires only:
1. Create src/agents/new_agent.py with a class inheriting BaseAgent.
2. Add one entry to REGISTRY below.
"""

from __future__ import annotations

from typing import Type

from .base_agent import BaseAgent
from .primary_root_agent import PrimaryRootAgent
from .primary_shoot_agent import PrimaryShootAgent
from .lateral_root_agent import LateralRootAgent
from .branch_agent import BranchAgent

REGISTRY: dict[str, Type[BaseAgent]] = {
    "primary_root":  PrimaryRootAgent,
    "primary_shoot": PrimaryShootAgent,
    "lateral_root":  LateralRootAgent,
    "branch":        BranchAgent,
}


def spawn(agent_type: str, **kwargs) -> BaseAgent:
    """Instantiate an agent by type string.

    Args:
        agent_type: One of the keys in REGISTRY.
        **kwargs:   Forwarded to the agent constructor.

    Raises:
        ValueError: If the agent_type is not in the registry.
    """
    if agent_type not in REGISTRY:
        raise ValueError(
            f"Unknown agent_type '{agent_type}'. "
            f"Valid types: {list(REGISTRY.keys())}"
        )
    return REGISTRY[agent_type](**kwargs)


def registered_types() -> list[str]:
    """Return a sorted list of all registered agent type strings."""
    return sorted(REGISTRY.keys())
