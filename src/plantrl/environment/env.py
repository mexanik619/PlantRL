"""PlantEnv — Gymnasium environment for the Plant RL simulation.

Sequential multi-agent system: one meristem tip active at a time, others in FIFO queue.
All tips share a single SAC policy via shared observation/action spaces.

Observation: Box(28,) float32  (see agents/base_agent.py for layout)
Action:      Box(1,)  float32  ∈ [-0.35, 0.35] radians
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from plantrl.agents.agent_registry import spawn
from plantrl.agents.types import (
    AgentHandoff,
    GlobalPlantState,
    GrowthSegment,
    StopReason,
    WorldSnapshot,
)
from plantrl.agents.base_agent import OBS_DIM
from plantrl.world.light_field import LightField
from plantrl.world.moisture_field import MoistureField
from plantrl.world.hormone_field import HormoneField

# ──────────────────────────────────────────────
# World geometry constants
# ──────────────────────────────────────────────
WORLD_WIDTH = 30.0        # ±15 x
WORLD_HEIGHT = 15.0       # above soil
WORLD_DEPTH = 12.0        # below soil (y negative)
SOIL_BOUNDARY = 0.0       # y = 0 separates above/below ground

# Branch spawning milestones
ROOT_LATERAL_DEPTH = -3.0      # primary root must reach this depth to spawn laterals
SHOOT_BRANCH_HEIGHT = 3.0      # primary shoot must reach this height to spawn branches

# Max agents per episode (prevents queue explosion)
MAX_TOTAL_AGENTS = 20

# Normalisation
HALF_WIDTH = WORLD_WIDTH / 2.0
Y_RANGE = WORLD_HEIGHT + WORLD_DEPTH   # total vertical span


class PlantEnv(gym.Env):
    """Gymnasium environment for the RL plant growth simulation.

    Curriculum stages (set via ``env.stage``):
        1 — fixed light, single moisture pocket, no obstacles, no gravity tilt
        2 — random light/moisture positions, simple obstacles, ±15° gravity
        3 — moving light, depleting soil, dense obstacles, ±30° gravity mid-episode

    Args:
        stage:      Initial curriculum stage (1–3).
        max_agents: Maximum number of agent activations per episode.
        seed:       Optional RNG seed.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        stage: int = 1,
        max_agents: int = MAX_TOTAL_AGENTS,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        assert 1 <= stage <= 3, "stage must be 1, 2, or 3"
        self.stage = stage
        self.max_agents = max_agents
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.ones(OBS_DIM, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-0.35], dtype=np.float32),
            high=np.array([0.35], dtype=np.float32),
            dtype=np.float32,
        )

        # World fields
        self.light = LightField(falloff_k=0.05)
        self.moisture = MoistureField(soil_boundary=SOIL_BOUNDARY)
        self.hormones = HormoneField(
            world_bounds=(-HALF_WIDTH, HALF_WIDTH, -WORLD_DEPTH, WORLD_HEIGHT)
        )

        # Episode state (initialised in reset)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._agent_queue: deque = deque()
        self._active_agent = None
        self._all_segments: list[GrowthSegment] = []
        self._completed_agents: list = []   # list of (agent, stop_reason)
        self._total_agents_spawned: int = 0
        self._gravity: tuple[float, float] = (0.0, -1.0)
        self._obstacles: list[tuple[float, float, float, float]] = []
        self._global_state: GlobalPlantState = GlobalPlantState()
        self._last_handoff: AgentHandoff = AgentHandoff()
        self._gravity_tilt_step: int = 0   # stage 3 mid-episode tilt step

        # Track whether milestone-based children have been spawned
        self._lateral_roots_spawned: bool = False
        self._branches_spawned: bool = False
        self._episode_done: bool = False

    def set_stage(self, stage: int) -> None:
        """Switch curriculum stage; takes effect at the next ``reset()``.

        Call it through ``vec_env.env_method("set_stage", n)``: ``VecEnv.set_attr``
        only sets the attribute on the outer Monitor wrapper, not on this env.
        """
        if not 1 <= stage <= 3:
            raise ValueError(f"stage must be 1, 2, or 3 (got {stage})")
        self.stage = stage

    # ──────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset episode state
        self._agent_queue = deque()
        self._active_agent = None
        self._all_segments = []
        self._completed_agents = []
        self._total_agents_spawned = 0
        self._lateral_roots_spawned = False
        self._branches_spawned = False
        self._episode_done = False
        self._last_handoff = AgentHandoff()
        self._global_state = GlobalPlantState()
        self._gravity_tilt_step = 0

        # Build world for this stage
        self._setup_world()

        # Spawn initial root + shoot
        self._enqueue_agent("primary_root", start_pos=(0.0, 0.0))
        self._enqueue_agent("primary_shoot", start_pos=(0.0, 0.0))

        # Begin with first agent
        self._activate_next_agent()
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Auto-reset guard: SubprocVecEnv may call step() before sending
        # reset() on an already-finished episode.  Return a fresh episode
        # instead of crashing.
        if self._episode_done:
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        delta_theta = float(np.clip(action[0], -0.35, 0.35))

        agent = self._active_agent
        if agent is None:
            # Queue exhausted — should not normally happen; auto-reset
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        # Sample world at current tip before the step
        world_snap = self._sample_world(agent.tip_pos)

        # Execute agent step
        new_pos, reward, agent_done, stop_reason = agent.step(
            delta_theta, world_snap, obstacles=self._obstacles
        )

        if new_pos is not None:
            self._all_segments.extend(agent.segments[-1:])  # append latest segment
            self._maybe_spawn_children(agent)

        # Update hormone + global state
        self._update_hormones()
        self._update_global_state()

        # Stage 3: dynamic updates
        if self.stage == 3:
            self.light.update(dt=1.0)
            self.moisture.deplete(dt=1.0)
            self._maybe_tilt_gravity()

        info: dict[str, Any] = {
            "stop_reason": stop_reason.name if agent_done else None,
            "agent_type": agent.agent_type,
            "steps_taken": agent.steps_taken,
        }

        if agent_done:
            self._completed_agents.append((agent, stop_reason))
            self._last_handoff = AgentHandoff(stop_reason=stop_reason)

            env_done = len(self._agent_queue) == 0
            if not env_done:
                self._activate_next_agent()
                obs = self._get_obs()
                return obs, reward, False, False, info
            else:
                self._episode_done = True
                obs = np.zeros(OBS_DIM, dtype=np.float32)
                return obs, reward, True, False, info

        obs = self._get_obs()
        return obs, reward, False, False, info

    # ──────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        agent = self._active_agent
        if agent is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        tip_x, tip_y = agent.tip_pos
        world_snap = self._sample_world((tip_x, tip_y))

        norm_x = tip_x / HALF_WIDTH
        norm_y = tip_y / (WORLD_HEIGHT if not agent.IS_ROOT else WORLD_DEPTH)

        return agent.build_obs(world_snap, self._global_state, norm_x, norm_y)

    # ──────────────────────────────────────────────
    # World sampling
    # ──────────────────────────────────────────────

    def _sample_world(self, pos: tuple[float, float]) -> WorldSnapshot:
        x, y = pos
        ldir = self.light.get_direction(x, y)
        lint = self.light.get_intensity(x, y)
        mgrad = self.moisture.get_gradient(x, y)
        mlev = self.moisture.get_level(x, y)
        gx, gy = self._gravity
        auxin = self.hormones.get_auxin(x, y)
        cyto = self.hormones.get_cytokinin(x, y)
        return WorldSnapshot(
            light_dir_x=ldir[0],
            light_dir_y=ldir[1],
            light_intensity=lint,
            moisture_grad_x=mgrad[0],
            moisture_grad_y=mgrad[1],
            moisture_level=mlev,
            gravity_x=gx,
            gravity_y=gy,
            auxin_local=auxin,
            cytokinin_local=cyto,
            water_target=self._nearest_water(pos),
            light_target=(float(self.light.source_x), float(self.light.source_y)),
        )

    def _nearest_water(self, pos: tuple[float, float]) -> tuple[float, float] | None:
        """Centre of the nearest moisture pocket that is not yet depleted (reward shaping only)."""
        live = [s for s in self.moisture.sources if s.peak >= 0.05]
        if not live:
            return None
        s = min(live, key=lambda s: (s.cx - pos[0]) ** 2 + (s.cy - pos[1]) ** 2)
        return (float(s.cx), float(s.cy))

    # ──────────────────────────────────────────────
    # Global plant state
    # ──────────────────────────────────────────────

    def _update_global_state(self) -> None:
        shoot_tips = [
            a for a, _ in self._completed_agents if not a.IS_ROOT
        ]
        root_tips = [
            a for a, _ in self._completed_agents if a.IS_ROOT
        ]
        if self._active_agent is not None:
            if not self._active_agent.IS_ROOT:
                shoot_tips.append(self._active_agent)
            else:
                root_tips.append(self._active_agent)

        # Shoot light deficit = avg (1 - intensity) for shoot tips
        if shoot_tips:
            deficit = np.mean([
                1.0 - self.light.get_intensity(*a.tip_pos)
                for a in shoot_tips
            ])
        else:
            deficit = 1.0

        # Root water = avg moisture at root tips
        if root_tips:
            water = np.mean([
                self.moisture.get_level(*a.tip_pos)
                for a in root_tips
            ])
        else:
            water = 0.0

        total_biomass = len(self._all_segments) / (self.max_agents * 50)  # rough norm

        # Height of highest shoot tip
        shoot_ys = [a.tip_pos[1] for a in shoot_tips] or [0.0]
        shoot_height_norm = max(shoot_ys) / WORLD_HEIGHT

        # Depth of deepest root tip
        root_ys = [a.tip_pos[1] for a in root_tips] or [0.0]
        root_depth_norm = abs(min(root_ys)) / WORLD_DEPTH

        self._global_state = GlobalPlantState(
            shoot_light_deficit=float(np.clip(deficit, 0.0, 1.0)),
            root_water_found=float(np.clip(water, 0.0, 1.0)),
            total_biomass_norm=float(np.clip(total_biomass, 0.0, 1.0)),
            shoot_height_norm=float(np.clip(shoot_height_norm, 0.0, 1.0)),
            root_depth_norm=float(np.clip(root_depth_norm, 0.0, 1.0)),
        )

    # ──────────────────────────────────────────────
    # Hormone update
    # ──────────────────────────────────────────────

    def _update_hormones(self) -> None:
        all_agents = (
            [a for a, _ in self._completed_agents]
            + ([self._active_agent] if self._active_agent else [])
        )
        auxin_sources = [a.tip_pos for a in all_agents if not a.IS_ROOT]
        cyto_sources = [a.tip_pos for a in all_agents if a.IS_ROOT]
        self.hormones.update(auxin_sources, cyto_sources)

    # ──────────────────────────────────────────────
    # Child agent spawning (milestone-based)
    # ──────────────────────────────────────────────

    def _maybe_spawn_children(self, agent) -> None:
        if self._total_agents_spawned >= self.max_agents:
            return

        tip_x, tip_y = agent.tip_pos

        # Primary root reaching depth milestone → spawn lateral roots
        if (
            agent.agent_type == "primary_root"
            and not self._lateral_roots_spawned
            and tip_y <= ROOT_LATERAL_DEPTH
        ):
            self._lateral_roots_spawned = True
            for angle_offset in [-math.pi / 4, math.pi / 4]:
                spawn_angle = agent.tip_angle + angle_offset
                self._enqueue_agent(
                    "lateral_root",
                    start_pos=(tip_x, tip_y),
                    parent_angle=spawn_angle,
                    handoff=self._last_handoff,
                )

        # Primary shoot reaching height milestone → spawn branches
        if (
            agent.agent_type == "primary_shoot"
            and not self._branches_spawned
            and tip_y >= SHOOT_BRANCH_HEIGHT
        ):
            self._branches_spawned = True
            for angle_offset in [-math.pi / 4, math.pi / 4]:
                spawn_angle = agent.tip_angle + angle_offset
                self._enqueue_agent(
                    "branch",
                    start_pos=(tip_x, tip_y),
                    parent_angle=spawn_angle,
                    handoff=self._last_handoff,
                )

    # ──────────────────────────────────────────────
    # Queue management
    # ──────────────────────────────────────────────

    def _enqueue_agent(self, agent_type: str, **kwargs) -> None:
        if self._total_agents_spawned >= self.max_agents:
            return
        kwargs.setdefault("handoff", self._last_handoff)
        agent = spawn(agent_type, **kwargs)
        self._agent_queue.append(agent)
        self._total_agents_spawned += 1

    def _activate_next_agent(self) -> None:
        if self._agent_queue:
            self._active_agent = self._agent_queue.popleft()
        else:
            self._active_agent = None

    # ──────────────────────────────────────────────
    # World setup per stage
    # ──────────────────────────────────────────────

    def _setup_world(self) -> None:
        stage = self.stage
        rng = self._rng

        # -- Light --
        if stage == 1:
            self.light = LightField(source_pos=(0.0, 10.0), moving=False)
        else:
            self.light = LightField(moving=(stage == 3))
            self.light.reset(rng=rng, world_width=WORLD_WIDTH, moving=(stage == 3))

        # -- Moisture --
        n_sources = {1: 1, 2: 2, 3: 3}[stage]
        self.moisture.reset(
            rng=rng,
            n_sources=n_sources,
            world_width=WORLD_WIDTH,
            max_depth=WORLD_DEPTH,
            depleting=(stage == 3),
        )

        # -- Hormones --
        self.hormones.reset()

        # -- Gravity --
        if stage == 1:
            self._gravity = (0.0, -1.0)
        elif stage == 2:
            tilt = float(rng.uniform(-15, 15)) * math.pi / 180.0
            self._gravity = (math.sin(tilt), -math.cos(tilt))
        else:
            tilt = float(rng.uniform(-30, 30)) * math.pi / 180.0
            self._gravity = (math.sin(tilt), -math.cos(tilt))
            self._gravity_tilt_step = int(rng.integers(30, 80))

        # -- Obstacles --
        self._obstacles = []
        if stage == 2:
            n_obs = int(rng.integers(1, 4))
            for _ in range(n_obs):
                ox = float(rng.uniform(-10, 8))
                oy = float(rng.uniform(-8, 6))
                ow = float(rng.uniform(1, 3))
                oh = float(rng.uniform(0.5, 2))
                self._obstacles.append((ox, oy, ow, oh))
        elif stage == 3:
            n_obs = int(rng.integers(4, 9))
            for _ in range(n_obs):
                ox = float(rng.uniform(-12, 10))
                oy = float(rng.uniform(-10, 8))
                ow = float(rng.uniform(1, 4))
                oh = float(rng.uniform(0.5, 2.5))
                self._obstacles.append((ox, oy, ow, oh))

    # ──────────────────────────────────────────────
    # Stage 3: mid-episode gravity tilt
    # ──────────────────────────────────────────────

    def _maybe_tilt_gravity(self) -> None:
        """Apply a random gravity tilt once per episode at a predetermined step."""
        total_steps = sum(a.steps_taken for a, _ in self._completed_agents)
        if (
            self._gravity_tilt_step > 0
            and total_steps >= self._gravity_tilt_step
        ):
            tilt = float(self._rng.uniform(-30, 30)) * math.pi / 180.0
            self._gravity = (math.sin(tilt), -math.cos(tilt))
            self._gravity_tilt_step = 0  # only tilt once

    # ──────────────────────────────────────────────
    # Convenience properties
    # ──────────────────────────────────────────────

    @property
    def all_segments(self) -> list[GrowthSegment]:
        """All recorded growth segments so far (for rendering)."""
        return self._all_segments

    @property
    def active_agent(self):
        return self._active_agent

    def __repr__(self) -> str:
        return (
            f"PlantEnv(stage={self.stage}, "
            f"queue={len(self._agent_queue)}, "
            f"segments={len(self._all_segments)})"
        )
