"""
grid.py
-------
Defines the city grid layout using numpy.
Each cell holds one of: EMPTY, ROAD, HIGHWAY, INTERSECTION.
"""

import numpy as np
import random

# --- Cell type constants ---
EMPTY        = 0
ROAD         = 1
HIGHWAY      = 2
INTERSECTION = 3

# Human-readable names (useful for debugging)
CELL_NAMES = {
    EMPTY:        "empty",
    ROAD:         "road",
    HIGHWAY:      "highway",
    INTERSECTION: "intersection",
}


class CityGrid:
    """
    Represents the city as a 2D numpy array.

    Coordinates:  grid[row, col]  →  row = y, col = x
    Access helper: self.get(x, y)  /  self.set(x, y, cell_type)
    """

    def __init__(self, width: int = 20, height: int = 20):
        self.width  = width
        self.height = height
        # All cells start as EMPTY
        self.grid = np.zeros((height, width), dtype=np.int8)

    # ------------------------------------------------------------------
    # Basic access helpers
    # ------------------------------------------------------------------

    def get(self, x: int, y: int) -> int:
        """Return cell type at (x, y). Returns EMPTY for out-of-bounds."""
        if self.in_bounds(x, y):
            return int(self.grid[y, x])
        return EMPTY

    def set(self, x: int, y: int, cell_type: int) -> None:
        """Set cell type at (x, y) if in bounds."""
        if self.in_bounds(x, y):
            self.grid[y, x] = cell_type

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_drivable(self, x: int, y: int) -> bool:
        """Cars can only drive on road, highway, or intersection tiles."""
        return self.get(x, y) in (ROAD, HIGHWAY, INTERSECTION)

    # ------------------------------------------------------------------
    # Neighbour / connectivity helpers
    # ------------------------------------------------------------------

    def neighbours(self, x: int, y: int):
        """Yield drivable 4-connected neighbours of (x, y)."""
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.is_drivable(nx, ny):
                yield (nx, ny)

    def all_drivable_tiles(self):
        """Return a list of (x, y) tuples for every drivable cell."""
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.is_drivable(x, y):
                    positions.append((x, y))
        return positions

    # ------------------------------------------------------------------
    # Pre-built city layouts
    # ------------------------------------------------------------------

    def build_grid_city(self, block_size: int = 4) -> None:
        """
        Create a simple grid-street city.
        Streets run every `block_size` cells in both axes.
        Intersections are placed where horizontal and vertical roads meet.
        """
        self.grid[:] = EMPTY  # reset

        road_cols = list(range(0, self.width,  block_size))
        road_rows = list(range(0, self.height, block_size))

        # Lay horizontal roads
        for row in road_rows:
            for x in range(self.width):
                self.set(x, row, ROAD)

        # Lay vertical roads
        for col in road_cols:
            for y in range(self.height):
                self.set(col, y, ROAD)

        # Mark intersections where roads cross
        for row in road_rows:
            for col in road_cols:
                self.set(col, row, INTERSECTION)

        # Upgrade every other horizontal road to highway
        for i, row in enumerate(road_rows):
            if i % 2 == 0:
                for x in range(self.width):
                    if self.get(x, row) == ROAD:  # don't overwrite intersections
                        self.set(x, row, HIGHWAY)

    def build_random_city(self, road_density: float = 0.3) -> None:
        """
        Scatter roads randomly, then mark obvious intersections.
        Useful for stress-testing the pathfinder.
        """
        self.grid[:] = EMPTY
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < road_density:
                    self.set(x, y, ROAD)

        # Any road tile with ≥3 drivable neighbours becomes an intersection
        for y in range(self.height):
            for x in range(self.width):
                if self.get(x, y) == ROAD:
                    nb = list(self.neighbours(x, y))
                    if len(nb) >= 3:
                        self.set(x, y, INTERSECTION)

    # ------------------------------------------------------------------
    # RL action helpers
    # ------------------------------------------------------------------

    def add_road(self, x: int, y: int) -> bool:
        """Place a ROAD tile if the cell is currently EMPTY. Returns True on success."""
        if self.get(x, y) == EMPTY:
            self.set(x, y, ROAD)
            return True
        return False

    def upgrade_to_highway(self, x: int, y: int) -> bool:
        """Upgrade ROAD → HIGHWAY. Returns True on success."""
        if self.get(x, y) == ROAD:
            self.set(x, y, HIGHWAY)
            return True
        return False

    def add_intersection(self, x: int, y: int) -> bool:
        """Upgrade ROAD or HIGHWAY → INTERSECTION. Returns True on success."""
        if self.get(x, y) in (ROAD, HIGHWAY):
            self.set(x, y, INTERSECTION)
            return True
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        symbols = {EMPTY: ".", ROAD: "+", HIGHWAY: "=", INTERSECTION: "X"}
        rows = []
        for y in range(self.height):
            row = " ".join(symbols.get(self.grid[y, x], "?") for x in range(self.width))
            rows.append(row)
        return "\n".join(rows)
