from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pygame

from grid import CityGrid, Tile
from traffic_sim import TrafficSimulation


WINDOW_TITLE = "AI City Planner - Traffic Prototype"


class Palette:
    BG_TOP = (198, 198, 193)
    BG_BOTTOM = (184, 184, 179)

    DISTRICT = (178, 178, 176)
    DISTRICT_ALT = (168, 168, 167)
    PARK = (160, 170, 158)
    MAP_GRID = (136, 136, 138)

    BUILDING = (151, 151, 156)
    BUILDING_ALT = (142, 142, 148)
    BUILDING_EDGE = (70, 70, 83)

    ROAD_ASPHALT = (66, 66, 76)
    ROAD_EDGE = (218, 218, 223)
    ROAD_MARK = (236, 236, 236)

    HIGHWAY_ASPHALT = (36, 36, 48)
    HIGHWAY_EDGE = (240, 240, 245)
    HIGHWAY_MARK = (252, 252, 252)

    INTERSECTION_FILL = (82, 82, 94)
    CONGESTION = (220, 96, 110)

    CAR = (245, 248, 252)
    CAR_GLOW = (220, 235, 255, 95)

    PANEL_BG = (42, 45, 55)
    PANEL_ACCENT = (130, 170, 220)
    HUD_TEXT = (238, 241, 248)
    HUD_MUTED = (178, 188, 206)
    HUD_HIGHLIGHT = (170, 220, 255)


GridPos = Tuple[int, int]
PointF = Tuple[float, float]


class TrafficVisualizer:
    def __init__(
        self,
        grid: CityGrid,
        sim: TrafficSimulation,
        cell_size: int = 18,
        update_hz: int = 22,
    ) -> None:
        self.grid = grid
        self.sim = sim
        self.cell_size = cell_size
        self.update_hz = update_hz

        self.width = grid.width * cell_size + 370
        self.height = grid.height * cell_size
        self.panel_x = grid.width * cell_size
        self.city_view_width = self.panel_x

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Segoe UI", 25, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI", 17, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 15)
        self.font_small = pygame.font.SysFont("Consolas", 13)

        self.elapsed = 0.0

        # Keep mild projection distortion so map is not perfectly robotic,
        # but avoid the overly wiggly look.
        self.warp_strength = 0.18
        self.jitter_strength = 0.03
        self.curve_bend_strength = 0.10

        self.node_positions = self._build_projected_nodes()
        self.road_segments = self._build_road_segments()
        self.segment_curves = self._build_segment_curves()
        self.block_points = self._build_block_points()
        self.buildings = self._build_buildings()
        self.world_w, self.world_h = self._compute_world_size()

        self.camera_zoom = 1.35
        self.camera_min_zoom = 0.45
        self.camera_max_zoom = 3.6
        self.camera_x = max(0.0, (self.world_w - self.city_view_width / self.camera_zoom) * 0.5)
        self.camera_y = max(0.0, (self.world_h - self.height / self.camera_zoom) * 0.56)
        self.dragging = False
        self.drag_last = (0, 0)

    def _build_projected_nodes(self) -> Dict[GridPos, PointF]:
        nodes: Dict[GridPos, PointF] = {}
        rng = np.random.default_rng(11)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                bx = x * self.cell_size + self.cell_size * 0.5
                by = (self.grid.height - 1 - y) * self.cell_size + self.cell_size * 0.5

                wx = (
                    bx
                    + math.sin(y * 0.22) * self.cell_size * self.warp_strength
                    + math.cos((x + y) * 0.11) * self.cell_size * (self.warp_strength * 0.55)
                )
                wy = (
                    by
                    + math.cos(x * 0.19) * self.cell_size * self.warp_strength
                    + math.sin((x - y) * 0.09) * self.cell_size * (self.warp_strength * 0.5)
                )
                jitter = rng.uniform(-self.jitter_strength, self.jitter_strength, size=2) * self.cell_size
                nodes[(x, y)] = (float(wx + jitter[0]), float(wy + jitter[1]))
        return nodes

    def _build_road_segments(self) -> List[Tuple[GridPos, GridPos, Tile]]:
        segments: List[Tuple[GridPos, GridPos, Tile]] = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                t1 = Tile(int(self.grid.grid[y, x]))
                if t1 not in (Tile.ROAD, Tile.HIGHWAY, Tile.INTERSECTION):
                    continue

                for nx, ny in ((x + 1, y), (x, y + 1)):
                    if not (0 <= nx < self.grid.width and 0 <= ny < self.grid.height):
                        continue
                    t2 = Tile(int(self.grid.grid[ny, nx]))
                    if t2 not in (Tile.ROAD, Tile.HIGHWAY, Tile.INTERSECTION):
                        continue

                    seg_tile = Tile.ROAD
                    if t1 == Tile.HIGHWAY or t2 == Tile.HIGHWAY:
                        seg_tile = Tile.HIGHWAY
                    elif t1 == Tile.INTERSECTION or t2 == Tile.INTERSECTION:
                        seg_tile = Tile.INTERSECTION
                    segments.append(((x, y), (nx, ny), seg_tile))
        return segments

    def _curve_points(self, p1: PointF, p2: PointF, seed: int) -> List[PointF]:
        x1, y1 = p1
        x2, y2 = p2
        vx = x2 - x1
        vy = y2 - y1
        seg_len = math.hypot(vx, vy)
        if seg_len <= 0.001:
            return [p1, p2]

        nx = -vy / seg_len
        ny = vx / seg_len

        # Nearly straight segments with very subtle bend.
        bend_base = (math.sin(seed * 0.71) * 0.38 + math.cos(seed * 1.17) * 0.22)
        bend = bend_base * self.cell_size * self.curve_bend_strength
        if seg_len < self.cell_size * 1.2:
            bend *= 0.25

        cx = (x1 + x2) * 0.5 + nx * bend
        cy = (y1 + y2) * 0.5 + ny * bend

        points: List[PointF] = []
        steps = 8
        for i in range(steps + 1):
            t = i / steps
            omt = 1.0 - t
            px = omt * omt * x1 + 2.0 * omt * t * cx + t * t * x2
            py = omt * omt * y1 + 2.0 * omt * t * cy + t * t * y2
            points.append((px, py))
        return points

    def _build_segment_curves(self) -> List[Tuple[List[PointF], Tile]]:
        curves: List[Tuple[List[PointF], Tile]] = []
        for a, b, tile in self.road_segments:
            p1 = self.node_positions[a]
            p2 = self.node_positions[b]
            seed = (a[0] * 73856093) ^ (a[1] * 19349663) ^ (b[0] * 83492791) ^ (b[1] * 2654435761)
            curves.append((self._curve_points(p1, p2, seed), tile))
        return curves

    def _build_block_points(self) -> List[Tuple[PointF, int, Tuple[int, int, int]]]:
        blocks: List[Tuple[PointF, int, Tuple[int, int, int]]] = []
        rng = np.random.default_rng(23)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                tile = Tile(int(self.grid.grid[y, x]))
                if tile != Tile.EMPTY:
                    continue
                if float(rng.random()) < 0.22:
                    p = self.node_positions[(x, y)]
                    size = int(rng.integers(2, 5))
                    color = Palette.DISTRICT if float(rng.random()) < 0.84 else Palette.DISTRICT_ALT
                    if float(rng.random()) < 0.07:
                        color = Palette.PARK
                        size = int(rng.integers(3, 6))
                    blocks.append((p, size, color))
        return blocks

    def _build_buildings(self) -> List[Tuple[PointF, float, float, float, Tuple[int, int, int]]]:
        """Create many small building boxes near streets for a city-map look."""
        buildings: List[Tuple[PointF, float, float, float, Tuple[int, int, int]]] = []
        rng = np.random.default_rng(101)

        for y in range(1, self.grid.height - 1):
            for x in range(1, self.grid.width - 1):
                tile = Tile(int(self.grid.grid[y, x]))
                if tile != Tile.EMPTY:
                    continue

                # Prefer parcels touching roads, like urban blocks.
                near_road = False
                for nx, ny in (
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                    (x + 1, y + 1),
                    (x - 1, y + 1),
                    (x + 1, y - 1),
                    (x - 1, y - 1),
                ):
                    if Tile(int(self.grid.grid[ny, nx])) in (Tile.ROAD, Tile.HIGHWAY, Tile.INTERSECTION):
                        near_road = True
                        break
                if not near_road or float(rng.random()) > 0.88:
                    continue

                center = self.node_positions[(x, y)]
                lots = int(rng.integers(1, 4))
                for _ in range(lots):
                    bw = float(rng.uniform(self.cell_size * 0.28, self.cell_size * 0.62))
                    bh = float(rng.uniform(self.cell_size * 0.22, self.cell_size * 0.58))
                    ox = float(rng.uniform(-self.cell_size * 0.34, self.cell_size * 0.34))
                    oy = float(rng.uniform(-self.cell_size * 0.34, self.cell_size * 0.34))
                    angle = float(rng.uniform(-0.35, 0.35))
                    color = Palette.BUILDING if float(rng.random()) < 0.6 else Palette.BUILDING_ALT
                    buildings.append(((center[0] + ox, center[1] + oy), bw, bh, angle, color))
        return buildings

    def _compute_world_size(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.node_positions.values()]
        ys = [p[1] for p in self.node_positions.values()]
        return max(xs) + self.cell_size * 2.2, max(ys) + self.cell_size * 2.2

    def _to_screen(self, world_pos: PointF) -> Tuple[int, int]:
        wx, wy = world_pos
        return int((wx - self.camera_x) * self.camera_zoom), int((wy - self.camera_y) * self.camera_zoom)

    def _clamp_camera(self) -> None:
        view_w = self.city_view_width / self.camera_zoom
        view_h = self.height / self.camera_zoom
        max_x = max(0.0, self.world_w - view_w)
        max_y = max(0.0, self.world_h - view_h)
        self.camera_x = min(max(self.camera_x, 0.0), max_x)
        self.camera_y = min(max(self.camera_y, 0.0), max_y)

    def _zoom_at(self, mouse_x: int, mouse_y: int, factor: float) -> None:
        if mouse_x >= self.city_view_width:
            return
        world_x = self.camera_x + mouse_x / self.camera_zoom
        world_y = self.camera_y + mouse_y / self.camera_zoom

        self.camera_zoom = float(np.clip(self.camera_zoom * factor, self.camera_min_zoom, self.camera_max_zoom))
        self.camera_x = world_x - mouse_x / self.camera_zoom
        self.camera_y = world_y - mouse_y / self.camera_zoom
        self._clamp_camera()

    def _draw_gradient_bg(self) -> None:
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            r = int(Palette.BG_TOP[0] * (1 - t) + Palette.BG_BOTTOM[0] * t)
            g = int(Palette.BG_TOP[1] * (1 - t) + Palette.BG_BOTTOM[1] * t)
            b = int(Palette.BG_TOP[2] * (1 - t) + Palette.BG_BOTTOM[2] * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.city_view_width, y))

        # Cartographic sheet grid overlay.
        spacing = int(self.cell_size * 4.5 * self.camera_zoom)
        spacing = max(28, spacing)
        for x in range(0, self.city_view_width, spacing):
            pygame.draw.line(self.screen, Palette.MAP_GRID, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, spacing):
            pygame.draw.line(self.screen, Palette.MAP_GRID, (0, y), (self.city_view_width, y), 1)

    def _draw_districts(self) -> None:
        for wp, size, color in self.block_points:
            sx, sy = self._to_screen(wp)
            r = max(1, int(size * self.camera_zoom * 0.45))
            pygame.draw.rect(self.screen, color, pygame.Rect(sx - r, sy - r, r * 2, r * 2), border_radius=max(1, r // 2))

        # Buildings as small boxes with outline.
        for center, bw, bh, angle, color in self.buildings:
            cx, cy = self._to_screen(center)
            hw = max(1.5, bw * self.camera_zoom * 0.5)
            hh = max(1.5, bh * self.camera_zoom * 0.5)

            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            ca = math.cos(angle)
            sa = math.sin(angle)
            pts = []
            for x, y in corners:
                rx = x * ca - y * sa
                ry = x * sa + y * ca
                pts.append((int(cx + rx), int(cy + ry)))

            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, Palette.BUILDING_EDGE, pts, 1)

    def _draw_polyline(self, color: Tuple[int, int, int], points: List[PointF], width: int) -> None:
        if len(points) < 2:
            return
        screen_points = [self._to_screen(p) for p in points]
        pygame.draw.lines(self.screen, color, False, screen_points, width)

    def _draw_dashed_polyline(
        self,
        color: Tuple[int, int, int],
        points: List[PointF],
        dash_len: float,
        gap_len: float,
        width: int,
        phase: float,
    ) -> None:
        if len(points) < 2:
            return

        offset = phase % (dash_len + gap_len)
        draw_on = offset < dash_len
        remaining = (dash_len - offset) if draw_on else (dash_len + gap_len - offset)

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg = math.hypot(dx, dy)
            if seg <= 0.001:
                continue
            ux = dx / seg
            uy = dy / seg
            travelled = 0.0
            while travelled < seg:
                step = min(remaining, seg - travelled)
                if draw_on:
                    sx1 = x1 + ux * travelled
                    sy1 = y1 + uy * travelled
                    sx2 = x1 + ux * (travelled + step)
                    sy2 = y1 + uy * (travelled + step)
                    pygame.draw.line(
                        self.screen,
                        color,
                        self._to_screen((sx1, sy1)),
                        self._to_screen((sx2, sy2)),
                        width,
                    )

                travelled += step
                remaining -= step
                if remaining <= 0.001:
                    draw_on = not draw_on
                    remaining = dash_len if draw_on else gap_len

    def _draw_roads(self) -> None:
        for points, tile in self.segment_curves:
            if tile == Tile.HIGHWAY:
                asphalt = Palette.HIGHWAY_ASPHALT
                edge = Palette.HIGHWAY_EDGE
                mark = Palette.HIGHWAY_MARK
                base_w = max(4, int(self.cell_size * 0.34 * self.camera_zoom))
            elif tile == Tile.INTERSECTION:
                asphalt = Palette.INTERSECTION_FILL
                edge = Palette.ROAD_EDGE
                mark = Palette.ROAD_MARK
                base_w = max(3, int(self.cell_size * 0.28 * self.camera_zoom))
            else:
                asphalt = Palette.ROAD_ASPHALT
                edge = Palette.ROAD_EDGE
                mark = Palette.ROAD_MARK
                base_w = max(3, int(self.cell_size * 0.26 * self.camera_zoom))

            edge_w = base_w + max(2, int(self.camera_zoom))
            self._draw_polyline(edge, points, edge_w)
            self._draw_polyline(asphalt, points, base_w)

            # Static lane marks (no animation).
            lane_w = max(1, int(self.camera_zoom))
            self._draw_dashed_polyline(mark, points, 8.0, 7.0, lane_w, 0.0)
            if tile == Tile.HIGHWAY:
                self._draw_dashed_polyline(Palette.HIGHWAY_MARK, points, 14.0, 11.0, lane_w, 0.0)

        # Smooth intersection caps over junction points.
        for (x, y), wp in self.node_positions.items():
            tile = Tile(int(self.grid.grid[y, x]))
            if tile not in (Tile.ROAD, Tile.HIGHWAY, Tile.INTERSECTION):
                continue
            p = self._to_screen(wp)
            r = max(2, int(self.cell_size * 0.12 * self.camera_zoom))
            color = Palette.ROAD_ASPHALT
            if tile == Tile.HIGHWAY:
                color = Palette.HIGHWAY_ASPHALT
                r += 1
            elif tile == Tile.INTERSECTION:
                color = Palette.INTERSECTION_FILL
                r += 1
            pygame.draw.circle(self.screen, color, p, r)
            pygame.draw.circle(self.screen, Palette.ROAD_EDGE, p, r + 1, 1)

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.update_hz) / 1000.0
            self.elapsed += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if event.y > 0:
                        self._zoom_at(mx, my, 1.16)
                    elif event.y < 0:
                        self._zoom_at(mx, my, 0.86)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if event.pos[0] < self.city_view_width:
                        self.dragging = True
                        self.drag_last = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging = False
                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    dx = event.pos[0] - self.drag_last[0]
                    dy = event.pos[1] - self.drag_last[1]
                    self.camera_x -= dx / self.camera_zoom
                    self.camera_y -= dy / self.camera_zoom
                    self.drag_last = event.pos

            keys = pygame.key.get_pressed()
            pan_speed = 17.0 / self.camera_zoom
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.camera_x -= pan_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.camera_x += pan_speed
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.camera_y -= pan_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.camera_y += pan_speed
            if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                self._zoom_at(self.city_view_width // 2, self.height // 2, 1.02)
            if keys[pygame.K_MINUS]:
                self._zoom_at(self.city_view_width // 2, self.height // 2, 0.98)

            self._clamp_camera()
            self.sim.step()
            self.draw()
            pygame.display.flip()

        pygame.quit()

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))
        self._draw_gradient_bg()
        self._draw_districts()
        self._draw_roads()

        density = self.sim.traffic_density_map()
        max_density = float(np.max(density)) if density.size > 0 else 0.0

        glow_surface = pygame.Surface((self.city_view_width, self.height), pygame.SRCALPHA)
        if max_density > 0:
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    d = float(density[y, x])
                    if d <= 0.0:
                        continue
                    p = self._to_screen(self.node_positions[(x, y)])
                    alpha = min(145, int((d / max_density) * 110) + 24)
                    pygame.draw.circle(
                        glow_surface,
                        (*Palette.CONGESTION, alpha),
                        p,
                        max(3, int(self.cell_size * 0.24 * self.camera_zoom)),
                    )
        self.screen.blit(glow_surface, (0, 0))

        car_glow = pygame.Surface((self.city_view_width, self.height), pygame.SRCALPHA)
        for car in self.sim.cars.values():
            px, py = self._to_screen(self.node_positions[car.position])
            glow_r = max(2, int(2.4 * self.camera_zoom))
            dot_r = max(1, int(1.2 * self.camera_zoom))
            pygame.draw.circle(car_glow, Palette.CAR_GLOW, (px, py), glow_r)
            pygame.draw.circle(self.screen, Palette.CAR, (px, py), dot_r)
        self.screen.blit(car_glow, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

        self.draw_side_panel()

    def draw_side_panel(self) -> None:
        panel_rect = pygame.Rect(self.panel_x, 0, self.width - self.panel_x, self.height)
        pygame.draw.rect(self.screen, Palette.PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, Palette.PANEL_ACCENT, (self.panel_x, 0), (self.panel_x, self.height), 2)

        for y in range(30, self.height, 84):
            pygame.draw.line(
                self.screen,
                (38, 54, 74),
                (self.panel_x + 16, y),
                (self.width - 16, y),
                1,
            )

        metrics = self.sim.metrics()
        lines = [
            "AI CITY PLANNER",
            "Modern Cartographic View",
            "",
            f"Tick: {int(metrics['tick'])}",
            f"Active Cars: {int(metrics['cars_active'])}",
            f"Completed Trips: {int(metrics['cars_completed'])}",
            f"Avg Travel: {metrics['average_travel_time']:.2f}",
            f"Stopped: {int(metrics['stopped_cars'])}",
            f"Congestion: {metrics['congestion']:.3f}",
            "",
            "Controls:",
            "Wheel or +/- : Zoom",
            "Drag or WASD : Pan",
            "",
            "Road Legend:",
            "Dashed white = lane marks",
            "Small boxes = buildings",
        ]

        top = 22
        for idx, text in enumerate(lines):
            is_heading = idx == 0
            is_subtitle = idx == 1
            is_metric = 3 <= idx <= 7

            color = Palette.HUD_TEXT
            if is_metric:
                color = Palette.HUD_HIGHLIGHT
            elif not is_heading and not is_subtitle:
                color = Palette.HUD_MUTED

            font = self.font_body
            if is_heading:
                font = self.font_title
            elif is_subtitle:
                font = self.font_subtitle

            surf = font.render(text, True, color)
            self.screen.blit(surf, (self.panel_x + 20, top + idx * 28))

        footer = f"FPS {int(self.clock.get_fps())}  |  Zoom {self.camera_zoom:.2f}x"
        info = self.font_small.render(footer, True, (170, 188, 214))
        self.screen.blit(info, (self.panel_x + 20, self.height - 26))


def run_visualization(grid: CityGrid, sim: TrafficSimulation) -> None:
    pygame.init()
    app = TrafficVisualizer(grid, sim)
    app.run()
