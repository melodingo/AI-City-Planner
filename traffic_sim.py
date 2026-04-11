"""
traffic_sim.py
--------------
Car class, A* pathfinding, and the SimulationEngine that runs discrete ticks.

Key design decisions
--------------------
- One car occupies exactly one tile at a time.
- A car "owns" its current position in the engine's occupancy set so that
  other cars can check for collisions in O(1).
- Intersections act as single-slot queues: only one car may enter per tick.
"""

import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from grid import CityGrid, EMPTY, ROAD, HIGHWAY, INTERSECTION

# ---------------------------------------------------------------------------
# Movement cost per tile type (lower = faster road)
# ---------------------------------------------------------------------------
MOVE_COST = {
    ROAD:         2,   # normal road: slower
    HIGHWAY:      1,   # highway:     faster
    INTERSECTION: 2,   # treat like road for pathfinding
}


# ---------------------------------------------------------------------------
# A* pathfinder
# ---------------------------------------------------------------------------

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance – admissible heuristic for 4-connected grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: CityGrid,
          start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Return the shortest drivable path from start to goal, or None if
    no path exists.  Path includes both endpoints.
    """
    if start == goal:
        return [start]

    # open_set entries: (f_score, g_score, position)
    open_set: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))

    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score:   Dict[Tuple[int, int], int] = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            node: Optional[Tuple[int, int]] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        # Skip stale open-set entries
        if g > g_score.get(current, float("inf")):
            continue

        for nb in grid.neighbours(*current):
            cost     = MOVE_COST.get(grid.get(*nb), 2)
            new_g    = g_score[current] + cost
            if new_g < g_score.get(nb, float("inf")):
                g_score[nb]  = new_g
                came_from[nb] = current
                f             = new_g + heuristic(nb, goal)
                heapq.heappush(open_set, (f, new_g, nb))

    return None  # no path found


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

@dataclass
class Car:
    """
    A single vehicle navigating the grid.

    Attributes
    ----------
    car_id      : unique integer identifier
    position    : current (x, y) tile
    destination : target (x, y) tile
    path        : remaining waypoints (index 0 = next step)
    travel_time : ticks elapsed since spawning
    stopped     : True when the car cannot move this tick
    waiting_at_intersection : ticks spent queuing at intersections
    """
    car_id:      int
    position:    Tuple[int, int]
    destination: Tuple[int, int]
    path:        List[Tuple[int, int]] = field(default_factory=list)

    travel_time: int  = 0
    stopped:     bool = False
    waiting_at_intersection: int = 0

    def has_arrived(self) -> bool:
        return self.position == self.destination

    def next_step(self) -> Optional[Tuple[int, int]]:
        """Return the next tile in the path without consuming it."""
        return self.path[0] if self.path else None

    def advance(self) -> None:
        """Move to the next path tile (call only after clearing collisions)."""
        if self.path:
            self.position = self.path.pop(0)
        self.travel_time += 1

    def tick_stopped(self) -> None:
        """Record a stopped tick without moving."""
        self.stopped = True
        self.travel_time += 1


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Manages the collection of cars and advances the simulation one tick at a
    time.

    Metrics tracked
    ---------------
    - average_travel_time : mean ticks across all active + completed cars
    - stopped_cars        : count of cars that could not move last tick
    - congestion_map      : numpy array, cars-per-tile at current tick
    """

    def __init__(self, grid: CityGrid,
                 max_cars: int = 30,
                 spawn_rate: float = 0.3):
        """
        Parameters
        ----------
        grid        : CityGrid instance
        max_cars    : cap on simultaneous cars
        spawn_rate  : probability of spawning a new car each tick
                      (only if below max_cars)
        """
        import numpy as np
        self.grid       = grid
        self.max_cars   = max_cars
        self.spawn_rate = spawn_rate

        self.cars:      List[Car] = []
        self.completed: List[Car] = []   # cars that reached destination
        self._next_id:  int       = 0
        self.tick_count: int      = 0

        # Set of occupied positions for O(1) collision checks
        self._occupied: set = set()

        # Intersection "locks": at most one car enters per intersection per tick
        self._intersection_locks: set = set()

        self.np = np

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cars and counters."""
        self.cars.clear()
        self.completed.clear()
        self._occupied.clear()
        self._next_id  = 0
        self.tick_count = 0

    def step(self) -> Dict:
        """
        Advance simulation by one tick.
        Returns a metrics dict.
        """
        self.tick_count += 1
        self._intersection_locks.clear()

        # Mark all cars as not-stopped at start of tick
        for car in self.cars:
            car.stopped = False

        # Move cars (shuffle to avoid systematic bias)
        order = list(range(len(self.cars)))
        random.shuffle(order)

        for i in order:
            if i >= len(self.cars):
                continue
            car = self.cars[i]
            self._try_move(car)

        # Remove arrived cars
        arrived = [c for c in self.cars if c.has_arrived()]
        for c in arrived:
            self._occupied.discard(c.position)
            self.completed.append(c)
        self.cars = [c for c in self.cars if not c.has_arrived()]

        # Spawn new cars
        self._maybe_spawn()

        return self.metrics()

    def metrics(self) -> Dict:
        """Return a snapshot of current simulation metrics."""
        all_cars = self.cars + self.completed
        avg_travel = (sum(c.travel_time for c in all_cars) / len(all_cars)
                      if all_cars else 0.0)

        stopped = sum(1 for c in self.cars if c.stopped)

        congestion = self.np.zeros(
            (self.grid.height, self.grid.width), dtype=np.float32
        )
        for c in self.cars:
            x, y = c.position
            congestion[y, x] += 1

        return {
            "tick":             self.tick_count,
            "active_cars":      len(self.cars),
            "completed_cars":   len(self.completed),
            "avg_travel_time":  round(avg_travel, 2),
            "stopped_cars":     stopped,
            "congestion_map":   congestion,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_move(self, car: Car) -> None:
        """Attempt to move a car one step along its path."""
        nxt = car.next_step()

        if nxt is None:
            # Path exhausted but not at destination → replan
            if not car.has_arrived():
                self._replan(car)
                nxt = car.next_step()

        if nxt is None:
            car.tick_stopped()
            return

        # Collision: another car is already on the next tile
        if nxt in self._occupied:
            car.tick_stopped()
            return

        # Intersection queue: only one car per intersection per tick
        if self.grid.get(*nxt) == INTERSECTION:
            if nxt in self._intersection_locks:
                car.tick_stopped()
                car.waiting_at_intersection += 1
                return
            self._intersection_locks.add(nxt)

        # All clear – move
        self._occupied.discard(car.position)
        self._occupied.add(nxt)
        car.advance()

    def _replan(self, car: Car) -> None:
        """Re-run A* for a car that lost its path."""
        path = astar(self.grid, car.position, car.destination)
        if path and len(path) > 1:
            car.path = path[1:]  # skip current position
        else:
            car.path = []

    def _maybe_spawn(self) -> None:
        """Randomly spawn a new car if below max_cars."""
        if len(self.cars) >= self.max_cars:
            return
        if random.random() > self.spawn_rate:
            return

        drivable = self.grid.all_drivable_tiles()
        if len(drivable) < 2:
            return

        # Pick a free spawn tile
        attempts = 0
        while attempts < 10:
            start = random.choice(drivable)
            if start not in self._occupied:
                break
            attempts += 1
        else:
            return

        # Pick a distinct destination
        dest_candidates = [t for t in drivable if t != start]
        if not dest_candidates:
            return
        dest = random.choice(dest_candidates)

        path = astar(self.grid, start, dest)
        if path is None or len(path) < 2:
            return  # no valid route

        car = Car(
            car_id      = self._next_id,
            position    = start,
            destination = dest,
            path        = path[1:],   # first step already "occupied"
        )
        self._next_id += 1
        self.cars.append(car)
        self._occupied.add(start)


import numpy as np  # make np available at module level for metrics()
