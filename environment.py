"""
environment.py
--------------
A Gym-style environment wrapping the city traffic simulation.
Designed so a Stable-Baselines3 agent (e.g. PPO) can be plugged in with
minimal changes later.

Observation space
-----------------
  Shape: (3, H, W)  – three "image" channels stacked
    [0] grid layout   (normalised cell type: 0-1)
    [1] traffic density per tile (0-1, clipped)
    [2] intersection map (binary)

Action space  (Discrete: 3 action types × grid cells)
--------------
  Encoded as a single integer:  action = action_type * (H*W) + tile_index
    action_type 0 → add_road       at tile (x,y)
    action_type 1 → upgrade_highway at tile (x,y)
    action_type 2 → add_intersection at tile (x,y)

Reward
------
    Each step uses *metric deltas* so improvements are rewarded:
        +w1 × (prev_avg_travel - avg_travel)
        +w2 × (prev_stopped - stopped)
        +w3 × (prev_mean_congestion - mean_congestion)
        +w4 × newly_completed_cars
        -build penalty (if action changed the grid)
        -invalid-action penalty (if action does nothing)
"""

import numpy as np
from grid import CityGrid, EMPTY, ROAD, HIGHWAY, INTERSECTION
from traffic_sim import SimulationEngine

# Max density value for normalisation (cars/tile)
MAX_DENSITY = 5.0


class TrafficEnv:
    """
    Gym-compatible environment for the city traffic simulation.

    Usage
    -----
    env = TrafficEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    """

    # Reward shaping weights for delta-based reward.
    W_DELTA_TRAVEL     = 1.40
    W_DELTA_STOPPED    = 0.90
    W_DELTA_CONGESTION = 3.20
    W_THROUGHPUT       = 0.80
    W_BUILD_PENALTY    = 0.05
    W_INVALID_ACTION   = 0.02

    def __init__(self,
                 width: int = 20,
                 height: int = 20,
                 max_cars: int = 30,
                 spawn_rate: float = 0.3,
                 episode_length: int = 200):

        self.width          = width
        self.height         = height
        self.episode_length = episode_length

        # --- observation / action dimensions ---
        self.n_channels  = 3
        self.obs_shape   = (self.n_channels, height, width)
        self.n_actions   = 3 * height * width   # 3 action types × every tile

        # --- internals ---
        self.grid   = CityGrid(width, height)
        self.engine = SimulationEngine(self.grid,
                                       max_cars=max_cars,
                                       spawn_rate=spawn_rate)
        self._t      = 0
        self._last_metrics: dict = {}
        self._prev_metrics: dict = {}

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment to a fresh grid city.
        Returns the initial observation.
        """
        self.grid.build_grid_city(block_size=4)
        self.engine.reset()
        self._t = 0
        # Run one warm-up tick so cars exist from the first observation
        self._last_metrics = self.engine.step()
        self._prev_metrics = dict(self._last_metrics)
        return self._build_observation()

    def step(self, action: int):
        """
        Apply an action, advance the simulation one tick, return
        (observation, reward, done, info).

        Parameters
        ----------
        action : int  –  encoded action (see module docstring)
        """
        # --- decode action ---
        action_type = action // (self.height * self.width)
        tile_index  = action  % (self.height * self.width)
        tile_y      = tile_index // self.width
        tile_x      = tile_index  % self.width

        grid_changed = self._apply_action(action_type, tile_x, tile_y)

        # --- advance simulation ---
        self._prev_metrics = dict(self._last_metrics)
        self._last_metrics = self.engine.step()
        self._t += 1

        # --- compute reward ---
        reward = self._compute_reward(grid_changed)

        # --- check termination ---
        done = self._t >= self.episode_length

        obs  = self._build_observation()
        info = {**self._last_metrics, "grid_changed": grid_changed}

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action_type: int, x: int, y: int) -> bool:
        """
        Apply one of the three grid-modification actions.
        Returns True if the grid was actually modified.
        """
        if action_type == 0:
            return self.grid.add_road(x, y)
        elif action_type == 1:
            return self.grid.upgrade_to_highway(x, y)
        elif action_type == 2:
            return self.grid.add_intersection(x, y)
        return False

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """
        Construct the (3, H, W) observation array.
        Channel 0: normalised cell type (0=empty … 3=intersection → /3)
        Channel 1: normalised traffic density (clipped to [0, 1])
        Channel 2: binary intersection mask
        """
        obs = np.zeros(self.obs_shape, dtype=np.float32)

        # Channel 0 – grid layout
        obs[0] = self.grid.grid.astype(np.float32) / 3.0

        # Channel 1 – traffic density
        cmap = self._last_metrics.get(
            "congestion_map",
            np.zeros((self.height, self.width), dtype=np.float32)
        )
        obs[1] = np.clip(cmap / MAX_DENSITY, 0.0, 1.0)

        # Channel 2 – intersection mask
        obs[2] = (self.grid.grid == INTERSECTION).astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, grid_changed: bool) -> float:
        prev = self._prev_metrics
        curr = self._last_metrics

        prev_travel = float(prev.get("avg_travel_time", 0.0))
        curr_travel = float(curr.get("avg_travel_time", 0.0))

        prev_stopped = float(prev.get("stopped_cars", 0.0))
        curr_stopped = float(curr.get("stopped_cars", 0.0))

        prev_cong = float(np.mean(prev.get("congestion_map", np.zeros(1))))
        curr_cong = float(np.mean(curr.get("congestion_map", np.zeros(1))))

        prev_completed = float(prev.get("completed_cars", 0.0))
        curr_completed = float(curr.get("completed_cars", 0.0))

        delta_travel = prev_travel - curr_travel
        delta_stopped = prev_stopped - curr_stopped
        delta_cong = prev_cong - curr_cong
        throughput = max(0.0, curr_completed - prev_completed)

        reward = 0.0
        reward += self.W_DELTA_TRAVEL * delta_travel
        reward += self.W_DELTA_STOPPED * delta_stopped
        reward += self.W_DELTA_CONGESTION * delta_cong
        reward += self.W_THROUGHPUT * throughput

        if grid_changed:
            reward -= self.W_BUILD_PENALTY
        else:
            reward -= self.W_INVALID_ACTION

        return float(reward)

    # ------------------------------------------------------------------
    # Convenience properties (for external tools / SB3)
    # ------------------------------------------------------------------

    @property
    def observation_space_shape(self):
        """Shape tuple – use to build a gym.spaces.Box later."""
        return self.obs_shape

    @property
    def action_space_n(self):
        """Number of discrete actions – use to build a gym.spaces.Discrete."""
        return self.n_actions

    def render_text(self) -> str:
        """Simple ASCII render for debugging."""
        m = self._last_metrics
        lines = [
            f"Tick {self._t}",
            f"  Active cars   : {m.get('active_cars', 0)}",
            f"  Completed     : {m.get('completed_cars', 0)}",
            f"  Avg travel    : {m.get('avg_travel_time', 0):.2f} ticks",
            f"  Stopped cars  : {m.get('stopped_cars', 0)}",
            "",
            str(self.grid),
        ]
        return "\n".join(lines)
