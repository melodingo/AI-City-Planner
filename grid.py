from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


Position = Tuple[int, int]


class Tile(IntEnum):
    EMPTY = 0
    ROAD = 1
    HIGHWAY = 2
    INTERSECTION = 3


DRIVABLE_TILES = {Tile.ROAD, Tile.HIGHWAY, Tile.INTERSECTION}


@dataclass
class CityGrid:
    width: int = 32
    height: int = 32

    def __post_init__(self) -> None:
        self.grid = np.full((self.height, self.width), Tile.EMPTY, dtype=np.int8)
        self._seed_basic_layout()

    def _seed_basic_layout(self) -> None:
        """Create a city-like network with ring roads, radials, and local streets."""
        self.grid[:, :] = Tile.EMPTY

        rng = np.random.default_rng(self.width * 1009 + self.height * 917)
        cx, cy = self.width / 2.0, self.height / 2.0
        rx = max(6.0, self.width * 0.38)
        ry = max(6.0, self.height * 0.34)

        # Two elliptical highway rings.
        self._draw_ring(cx, cy, rx, ry, thickness=0.12, tile=Tile.HIGHWAY)
        self._draw_ring(cx, cy, rx * 0.68, ry * 0.66, thickness=0.11, tile=Tile.HIGHWAY)

        # Radial arterials spread outward from core.
        radial_count = 14 if min(self.width, self.height) >= 40 else 10
        angle_jitter = float(rng.uniform(0.0, 2.0 * np.pi))
        for i in range(radial_count):
            angle = angle_jitter + i * (2.0 * np.pi / radial_count) + float(rng.uniform(-0.11, 0.11))
            self._draw_radial(cx, cy, angle, int(max(self.width, self.height) * 0.7))

        # Dense inner grid around downtown.
        inner_margin_x = max(4, int(self.width * 0.22))
        inner_margin_y = max(4, int(self.height * 0.22))
        for y in range(inner_margin_y, self.height - inner_margin_y, 3):
            self.grid[y, inner_margin_x : self.width - inner_margin_x] = Tile.ROAD
        for x in range(inner_margin_x, self.width - inner_margin_x, 3):
            self.grid[inner_margin_y : self.height - inner_margin_y, x] = Tile.ROAD

        # Organic neighborhood streets grown from random district seeds.
        district_count = 22 if min(self.width, self.height) >= 48 else 14
        for _ in range(district_count):
            sx = int(rng.integers(2, self.width - 2))
            sy = int(rng.integers(2, self.height - 2))
            self._grow_local_streets((sx, sy), rng, steps=int(max(self.width, self.height) * 2.4))

        self._promote_intersections()

    def _draw_ring(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        thickness: float,
        tile: Tile,
    ) -> None:
        for y in range(self.height):
            for x in range(self.width):
                nx = (x - cx) / max(1.0, rx)
                ny = (y - cy) / max(1.0, ry)
                d = np.sqrt(nx * nx + ny * ny)
                if abs(d - 1.0) <= thickness:
                    self.grid[y, x] = tile

    def _draw_radial(self, cx: float, cy: float, angle: float, length: int) -> None:
        for step in range(length):
            r = step * 0.75
            x = int(round(cx + np.cos(angle) * r + np.sin(step * 0.065) * 0.45))
            y = int(round(cy + np.sin(angle) * r + np.cos(step * 0.071) * 0.45))
            if not (0 <= x < self.width and 0 <= y < self.height):
                break
            self.grid[y, x] = Tile.HIGHWAY if step > length * 0.45 else Tile.ROAD

            # Give radial roads body width.
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == Tile.EMPTY:
                    self.grid[ny, nx] = Tile.ROAD

    def _grow_local_streets(self, start: Position, rng: np.random.Generator, steps: int) -> None:
        x, y = start
        direction = (1, 0)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(steps):
            if not (1 <= x < self.width - 1 and 1 <= y < self.height - 1):
                break
            if self.grid[y, x] == Tile.EMPTY:
                self.grid[y, x] = Tile.ROAD

            # Occasional lane branches.
            if i % 11 == 0:
                for bx, by in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 1 <= bx < self.width - 1 and 1 <= by < self.height - 1 and float(rng.random()) < 0.32:
                        self.grid[by, bx] = Tile.ROAD

            if float(rng.random()) < 0.18:
                direction = directions[int(rng.integers(0, len(directions)))]

            x += int(direction[0])
            y += int(direction[1])

    def _promote_intersections(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                tile = Tile(self.grid[y, x])
                if tile not in DRIVABLE_TILES:
                    continue
                neighbors = 0
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if Tile(self.grid[ny, nx]) in DRIVABLE_TILES:
                            neighbors += 1
                if neighbors >= 3:
                    self.grid[y, x] = Tile.INTERSECTION

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, pos: Position) -> Tile:
        x, y = pos
        return Tile(self.grid[y, x])

    def set_tile(self, pos: Position, tile: Tile) -> None:
        x, y = pos
        if self.in_bounds(pos):
            self.grid[y, x] = tile

    def is_drivable(self, pos: Position) -> bool:
        if not self.in_bounds(pos):
            return False
        return self.get_tile(pos) in DRIVABLE_TILES

    def neighbors4(self, pos: Position) -> List[Position]:
        x, y = pos
        candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        return [p for p in candidates if self.is_drivable(p)]

    def random_drivable_tile(self, rng: np.random.Generator) -> Optional[Position]:
        ys, xs = np.where(np.isin(self.grid, list(DRIVABLE_TILES)))
        if len(xs) == 0:
            return None
        idx = int(rng.integers(0, len(xs)))
        return int(xs[idx]), int(ys[idx])

    def drivable_count(self) -> int:
        return int(np.isin(self.grid, list(DRIVABLE_TILES)).sum())

    def as_numpy(self) -> np.ndarray:
        return self.grid.copy()
