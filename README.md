# AI City Planner

AI City Planner is a procedural city + traffic sandbox with two implementations:

- Browser simulation in city_visual.html (large canvas, animated traffic, UI controls).
- Python simulation stack (grid model, A* traffic engine, pygame renderer, Gym-style wrapper scaffold).

The project is focused on road hierarchy, connectivity, and how those choices impact congestion.

## Repository Layout

- city_visual.html: Browser-based city generation and traffic simulation.
- grid.py: City grid model and layout builders.
- traffic_sim.py: Car model, A* routing, and tick-by-tick simulation engine.
- environment.py: Gym-style environment wrapper around the simulation.
- visualize.py: Pygame renderer for the Python simulation.
- main.py: Command-line entry point for visual, headless, and env smoke-test modes.
- requirements.txt: Python dependencies.

## Browser Simulation (city_visual.html)

The browser version generates a full city and then runs continuous traffic.

### Road Classes

The current road hierarchy is:

- LOCAL: Neighborhood streets.
- ARTERIAL: Collector/distributor roads.
- HIGHWAY: Fast long-range roads.
- INTER: Intersection tiles that can mix classes.

### Generation Pipeline

The city is built in staged passes:

1. Boundary mask (irregular ellipse).
2. Zoning (CBD, commercial, residential, industrial, park).
3. Park placement.
4. Highways.
5. Arterials.
6. Local streets.
7. Cleanup (dead-end pruning, component filtering, intersection normalization).
8. Building placement.

### Buildings

Buildings are frontage-driven:

- Houses, towers, and warehouses are placed only on valid empty lots.
- Placement probability depends on zone type.
- Park zones are preserved as green tiles.

### Traffic Model Highlights

- Time-of-day demand profile (hourly spawn multipliers).
- A* pathfinding over the generated road network.
- Lane-aware rendering offsets and basic intersection locking behavior.
- Live metrics in HUD: tick, active cars, arrived cars, stopped cars, average trip, zoom.

### Browser Controls

- Speed: 0.5x, 1x, 4x.
- Pause/resume.
- Zoom in/out/reset fit.
- Mouse wheel zoom.
- Drag to pan.
- New City regeneration button.

## Python Simulation Stack

The Python side is a clean, modular simulation core for testing and experimentation.

### grid.py

CityGrid stores a 2D numpy grid with these cell types:

- EMPTY
- ROAD
- HIGHWAY
- INTERSECTION

Includes helper methods for bounds checks, drivable-neighbor queries, and two layout generators:

- build_grid_city(block_size=4)
- build_random_city(road_density=0.3)

### traffic_sim.py

Implements:

- A* pathfinding with Manhattan heuristic.
- Discrete car movement with occupancy checks.
- One-car-per-intersection-per-tick lock behavior.
- Spawn logic with max car cap.
- Metrics output per tick:
	- tick
	- active_cars
	- completed_cars
	- avg_travel_time
	- stopped_cars
	- congestion_map

### environment.py

TrafficEnv is a Gym-style wrapper (without hard dependency on gymnasium):

- Observation shape: (3, H, W)
	- Channel 0: normalized grid layout
	- Channel 1: normalized congestion map
	- Channel 2: intersection mask
- Action space size: 3 * H * W
	- action type 0: add road
	- action type 1: upgrade road to highway
	- action type 2: add intersection
- Reward uses weighted penalties for travel time, stopped cars, congestion, and build actions.

### visualize.py

Pygame renderer for the Python simulation:

- Draws base grid, congestion overlay, cars, and metric HUD.
- ESC or window close exits the render loop.

### main.py

CLI modes:

- visual (default): pygame window
- headless: text metrics in terminal
- envtest: random-action smoke test for TrafficEnv

## Installation

Requirements:

- Python 3.10+
- numpy>=1.24
- pygame>=2.5

Install:

```bash
pip install -r requirements.txt
```

If you only use city_visual.html in a browser, Python dependencies are optional.

## Usage

### Browser

Open city_visual.html in a modern browser.

### Python Visual Mode

```bash
python main.py
```

### Python Headless Mode

```bash
python main.py --mode headless --ticks 500
```

### Environment Smoke Test

```bash
python main.py --mode envtest --ticks 50
```

### Optional Headless Pygame CI/Smoke Runs

On systems without a display/audio device, pygame can run with dummy drivers:

```bash
set SDL_VIDEODRIVER=dummy
set SDL_AUDIODRIVER=dummy
python main.py --mode visual --ticks 50
```

## Current Limitations

- Browser and Python implementations are separate engines.
- RL integration is a scaffold, not a full training pipeline.
- Generated quality varies by random seed.

## License

No license file is currently included.
