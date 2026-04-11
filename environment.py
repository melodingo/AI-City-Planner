from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from grid import CityGrid, Position, Tile
from traffic_sim import TrafficSimulation


Action = Tuple[int, int, int]


@dataclass
class TrafficEnvConfig:
    width: int = 20
    height: int = 20
    max_steps: int = 1500


class CityTrafficEnv:
    """Gym-style wrapper for map-edit actions over the running traffic simulation."""

    def __init__(self, config: Optional[TrafficEnvConfig] = None, seed: int = 7) -> None:
        self.config = config or TrafficEnvConfig()
        self.seed = seed

        self.grid = CityGrid(width=self.config.width, height=self.config.height)
        self.sim = TrafficSimulation(self.grid, seed=self.seed)
        self.steps = 0

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "grid": self.grid.as_numpy().astype(np.int16),
            "density": self.sim.traffic_density_map(),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        self.grid = CityGrid(width=self.config.width, height=self.config.height)
        self.sim = TrafficSimulation(self.grid, seed=self.seed)
        self.steps = 0
        return self._observation()

    def _apply_action(self, action: Action) -> float:
        """Action format: (action_type, x, y). Returns build-cost penalty."""
        action_type, x, y = action
        pos: Position = (int(x), int(y))

        if not self.grid.in_bounds(pos):
            return -0.05

        tile = self.grid.get_tile(pos)

        if action_type == 0:
            # Add road.
            if tile == Tile.EMPTY:
                self.grid.set_tile(pos, Tile.ROAD)
                return -0.02
            return -0.005

        if action_type == 1:
            # Upgrade road to highway.
            if tile == Tile.ROAD:
                self.grid.set_tile(pos, Tile.HIGHWAY)
                return -0.03
            return -0.005

        if action_type == 2:
            # Add/force intersection.
            if tile in (Tile.ROAD, Tile.HIGHWAY):
                self.grid.set_tile(pos, Tile.INTERSECTION)
                return -0.02
            return -0.005

        return -0.01

    def step(self, action: Action):
        build_penalty = self._apply_action(action)
        self.sim.step()
        self.steps += 1

        metrics = self.sim.metrics()
        travel_penalty = -0.02 * metrics["average_travel_time"]
        congestion_penalty = -1.5 * metrics["congestion"]
        reward = travel_penalty + congestion_penalty + build_penalty

        done = self.steps >= self.config.max_steps
        info = metrics

        return self._observation(), float(reward), bool(done), info
