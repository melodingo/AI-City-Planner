from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from grid import CityGrid, Position, Tile


@dataclass
class Car:
    car_id: int
    position: Position
    destination: Position
    path: List[Position] = field(default_factory=list)
    spawn_tick: int = 0
    arrived_tick: Optional[int] = None
    stopped_ticks: int = 0


def _tile_movement_cost(tile: Tile) -> float:
    if tile == Tile.HIGHWAY:
        return 0.6
    if tile == Tile.INTERSECTION:
        return 1.1
    return 1.0


def _heuristic(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(grid: CityGrid, start: Position, goal: Position) -> List[Position]:
    """Return a path from start to goal (excluding start), or empty list if none."""
    if start == goal:
        return []
    if not grid.is_drivable(start) or not grid.is_drivable(goal):
        return []

    frontier: List[Tuple[float, Position]] = []
    heapq.heappush(frontier, (0.0, start))

    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost_so_far: Dict[Position, float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for nxt in grid.neighbors4(current):
            move_cost = _tile_movement_cost(grid.get_tile(nxt))
            new_cost = cost_so_far[current] + move_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return []

    path_rev = []
    cur = goal
    while cur != start:
        path_rev.append(cur)
        parent = came_from.get(cur)
        if parent is None:
            return []
        cur = parent
    path_rev.reverse()
    return path_rev


class TrafficSimulation:
    def __init__(
        self,
        grid: CityGrid,
        spawn_probability: float = 0.35,
        max_cars: int = 220,
        seed: int = 7,
    ) -> None:
        self.grid = grid
        self.spawn_probability = spawn_probability
        self.max_cars = max_cars
        self.rng = np.random.default_rng(seed)

        self.tick_count = 0
        self.next_car_id = 1

        self.cars: Dict[int, Car] = {}
        self.occupancy: Dict[Position, int] = {}

        self.completed_trip_count = 0
        self.total_trip_time = 0.0

        # Intersections may only accept one entering car per tick.
        self.intersection_used_this_tick: Set[Position] = set()

    def reset_runtime(self) -> None:
        self.tick_count = 0
        self.next_car_id = 1
        self.cars.clear()
        self.occupancy.clear()
        self.completed_trip_count = 0
        self.total_trip_time = 0.0

    def _pick_spawn_and_destination(self) -> Optional[Tuple[Position, Position, List[Position]]]:
        tries = 24
        for _ in range(tries):
            start = self.grid.random_drivable_tile(self.rng)
            goal = self.grid.random_drivable_tile(self.rng)
            if start is None or goal is None or start == goal:
                continue
            if start in self.occupancy:
                continue
            route = astar_path(self.grid, start, goal)
            if route:
                return start, goal, route
        return None

    def spawn_car_if_needed(self) -> None:
        if len(self.cars) >= self.max_cars:
            return
        if float(self.rng.random()) > self.spawn_probability:
            return

        choice = self._pick_spawn_and_destination()
        if not choice:
            return

        start, goal, route = choice
        car = Car(
            car_id=self.next_car_id,
            position=start,
            destination=goal,
            path=route,
            spawn_tick=self.tick_count,
        )
        self.cars[car.car_id] = car
        self.occupancy[start] = car.car_id
        self.next_car_id += 1

    def _can_enter(self, next_pos: Position) -> bool:
        tile = self.grid.get_tile(next_pos)
        if tile != Tile.INTERSECTION:
            return True
        if next_pos in self.intersection_used_this_tick:
            return False
        return True

    def _mark_enter(self, next_pos: Position) -> None:
        if self.grid.get_tile(next_pos) == Tile.INTERSECTION:
            self.intersection_used_this_tick.add(next_pos)

    def _advance_car(self, car: Car) -> None:
        if car.position == car.destination:
            return

        if not car.path:
            # Replan in case map changed.
            car.path = astar_path(self.grid, car.position, car.destination)
            if not car.path:
                car.stopped_ticks += 1
                return

        next_pos = car.path[0]

        # Stop if occupied by another car.
        if next_pos in self.occupancy:
            car.stopped_ticks += 1
            return

        # Intersection queue policy.
        if not self._can_enter(next_pos):
            car.stopped_ticks += 1
            return

        old_pos = car.position
        car.position = next_pos
        car.path.pop(0)

        self.occupancy.pop(old_pos, None)
        self.occupancy[next_pos] = car.car_id
        self._mark_enter(next_pos)

    def _complete_arrivals(self) -> None:
        arrived_ids: List[int] = []
        for car_id, car in self.cars.items():
            if car.position == car.destination:
                car.arrived_tick = self.tick_count
                self.total_trip_time += float(car.arrived_tick - car.spawn_tick)
                self.completed_trip_count += 1
                arrived_ids.append(car_id)

        for car_id in arrived_ids:
            pos = self.cars[car_id].position
            self.occupancy.pop(pos, None)
            self.cars.pop(car_id, None)

    def step(self) -> None:
        self.tick_count += 1
        self.intersection_used_this_tick.clear()

        self.spawn_car_if_needed()

        # Randomized update order avoids deterministic lane priority.
        car_ids = list(self.cars.keys())
        self.rng.shuffle(car_ids)
        for car_id in car_ids:
            car = self.cars.get(car_id)
            if car is not None:
                self._advance_car(car)

        self._complete_arrivals()

    def traffic_density_map(self) -> np.ndarray:
        density = np.zeros((self.grid.height, self.grid.width), dtype=np.float32)
        for car in self.cars.values():
            x, y = car.position
            density[y, x] += 1.0
        return density

    def metrics(self) -> Dict[str, float]:
        avg_travel_time = (
            self.total_trip_time / self.completed_trip_count
            if self.completed_trip_count > 0
            else 0.0
        )
        stopped = 0
        for car in self.cars.values():
            if car.path and car.path[0] in self.occupancy:
                stopped += 1

        drivable = max(1, self.grid.drivable_count())
        congestion = len(self.cars) / drivable

        return {
            "tick": float(self.tick_count),
            "cars_active": float(len(self.cars)),
            "cars_completed": float(self.completed_trip_count),
            "average_travel_time": float(avg_travel_time),
            "stopped_cars": float(stopped),
            "congestion": float(congestion),
        }
