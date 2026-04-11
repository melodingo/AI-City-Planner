"""
visualize.py
------------
Pygame-based renderer for the city traffic simulation.

Colour scheme
-------------
  EMPTY        → dark gray  (#1a1a2e)
  ROAD         → medium gray (#4a4a6a)
  HIGHWAY      → blue-gray   (#2d6a9f)
  INTERSECTION → teal        (#00b4d8)
  Car          → yellow dot  (#f4d35e)
  Congested    → red overlay (alpha-blended)
"""

import sys
import numpy as np

try:
    import pygame
except ImportError:
    print("pygame is not installed.  Run:  pip install pygame")
    sys.exit(1)

from grid import CityGrid, EMPTY, ROAD, HIGHWAY, INTERSECTION

# --- Visual constants ---
CELL_COLORS = {
    EMPTY:        (26,  26,  46),   # dark navy
    ROAD:         (74,  74, 106),   # muted purple-gray
    HIGHWAY:      (45, 106, 159),   # steel blue
    INTERSECTION: (0,  180, 216),   # teal
}
CAR_COLOR         = (244, 211,  94)  # golden yellow
CONGESTION_COLOR  = (220,  50,  50)  # red (used with alpha)
BACKGROUND_COLOR  = (15,  15,  30)   # near-black

# Minimum cells per pixel before the window becomes unwieldy
MIN_CELL_PX = 8


class CityRenderer:
    """
    Renders a CityGrid + live car positions using pygame.

    Parameters
    ----------
    grid        : CityGrid to render
    cell_size   : pixels per grid cell
    fps         : target frames per second
    show_congestion : overlay red tint on busy tiles
    """

    def __init__(self,
                 grid: CityGrid,
                 cell_size: int = 32,
                 fps: int = 10,
                 show_congestion: bool = True):

        self.grid             = grid
        self.cell_size        = max(cell_size, MIN_CELL_PX)
        self.fps              = fps
        self.show_congestion  = show_congestion

        self.screen_w = grid.width  * self.cell_size
        self.screen_h = grid.height * self.cell_size

        pygame.init()
        self.screen  = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock   = pygame.time.Clock()
        pygame.display.set_caption("City Traffic Simulation")

        # Surface for semi-transparent congestion overlay
        self._overlay = pygame.Surface(
            (self.cell_size, self.cell_size), pygame.SRCALPHA
        )

        # Font for HUD
        self._font = pygame.font.SysFont("monospace", 14)

    # ------------------------------------------------------------------
    # Main draw call
    # ------------------------------------------------------------------

    def draw(self,
             cars,
             metrics: dict,
             congestion_map: np.ndarray | None = None) -> bool:
        """
        Render one frame.

        Parameters
        ----------
        cars            : list of Car objects
        metrics         : dict returned by SimulationEngine.metrics()
        congestion_map  : optional (H, W) numpy array, cars per tile

        Returns False if the user closed the window.
        """
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(BACKGROUND_COLOR)

        # --- Draw grid tiles ---
        self._draw_grid()

        # --- Congestion overlay ---
        if self.show_congestion and congestion_map is not None:
            self._draw_congestion(congestion_map)

        # --- Draw cars ---
        self._draw_cars(cars)

        # --- HUD ---
        self._draw_hud(metrics)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        cs = self.cell_size
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell  = self.grid.get(x, y)
                color = CELL_COLORS.get(cell, CELL_COLORS[EMPTY])
                rect  = pygame.Rect(x * cs, y * cs, cs, cs)
                pygame.draw.rect(self.screen, color, rect)
                # thin grid lines
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def _draw_congestion(self, congestion_map: np.ndarray) -> None:
        cs        = self.cell_size
        max_cars  = max(congestion_map.max(), 1)

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                density = congestion_map[y, x]
                if density < 1:
                    continue
                alpha = int(min(density / max_cars, 1.0) * 160)
                self._overlay.fill((0, 0, 0, 0))  # clear
                self._overlay.fill((*CONGESTION_COLOR, alpha))
                self.screen.blit(self._overlay, (x * cs, y * cs))

    def _draw_cars(self, cars) -> None:
        cs     = self.cell_size
        radius = max(cs // 4, 3)

        for car in cars:
            cx = car.position[0] * cs + cs // 2
            cy = car.position[1] * cs + cs // 2
            # Stopped cars are drawn slightly smaller and in a different shade
            color = (200, 80, 80) if car.stopped else CAR_COLOR
            r     = radius - 1 if car.stopped else radius
            pygame.draw.circle(self.screen, color, (cx, cy), r)

    def _draw_hud(self, metrics: dict) -> None:
        """Small metrics overlay in the top-left corner."""
        lines = [
            f"Tick          : {metrics.get('tick', 0)}",
            f"Active cars   : {metrics.get('active_cars', 0)}",
            f"Completed     : {metrics.get('completed_cars', 0)}",
            f"Avg travel    : {metrics.get('avg_travel_time', 0):.1f} ticks",
            f"Stopped       : {metrics.get('stopped_cars', 0)}",
        ]
        pad = 6
        x, y = pad, pad
        for line in lines:
            # Shadow
            surf_shadow = self._font.render(line, True, (0, 0, 0))
            self.screen.blit(surf_shadow, (x + 1, y + 1))
            # Text
            surf = self._font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 2

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        pygame.quit()
