# AI City Planner

A city-generation and traffic-simulation project focused on turning simple grid logic into something that feels like a real place. The repository contains two related systems:

1. A browser-based city generator and live traffic visualizer in `city_visual.html`.
2. A Python city grid, traffic engine, and pygame renderer for simulation, testing, and reinforcement-learning experiments.

The project is built around the same core idea in both implementations: create a believable road network, place development where roads make sense, then measure how traffic behaves once the city is active.

## Highlights

- Generates a layered road hierarchy with highways, main roads, local roads, and intersections.
- Builds denser residential districts by favoring local-road frontage.
- Uses A* pathfinding so cars can route across the road network automatically.
- Tracks traffic metrics such as active cars, completed cars, stopped cars, and average travel time.
- Includes a closable tuning panel in the browser version so generation settings do not block the map.
- Supports both visual simulation and headless execution from Python.
- Provides a Gym-style environment wrapper that can be adapted for reinforcement learning.

## What This Project Is Trying to Model

The goal is not just to draw roads. It is to approximate the structure of a living city:

- Major roads organize large-scale movement.
- Main roads distribute traffic into districts.
- Local roads form the dense neighborhood network where housing is concentrated.
- Buildings should cluster along road frontage instead of filling the map randomly.
- Traffic should reveal how road hierarchy affects movement and congestion.

The browser version takes this idea further by stamping local-road districts and pushing housing density hard enough to create neighborhood-scale blocks rather than isolated buildings.

## Repository Layout

- `city_visual.html` - Full browser-based city generator, building placer, and traffic visualizer.
- `grid.py` - Grid representation and helper methods for the Python simulation.
- `traffic_sim.py` - A* routing, car behavior, and the simulation engine.
- `environment.py` - Gym-style environment wrapper for RL experiments.
- `visualize.py` - pygame renderer for the Python simulation.
- `main.py` - Command-line entry point for visual, headless, and environment test modes.
- `requirements.txt` - Python dependencies.

## Browser City Generator

The browser simulation in `city_visual.html` is the most visually ambitious part of the project. It renders a procedural city on a canvas and includes interactive controls for speed, zoom, and generation tuning.

### Road Hierarchy

The browser generator distinguishes four road classes:

- `LOCAL` - neighborhood streets and residential loops.
- `MAIN` - larger collectors that feed into highways.
- `HIGHWAY` - long-distance arterial corridors.
- `INTER` - intersections and junctions.

Road generation is intentionally hierarchical. Highways seed the network first, main roads branch off them, and local roads branch from the main network and from other local roads to create compact district patterns.

### Building Generation Approach

Buildings are not placed randomly across all empty tiles. They are only placed where frontage rules allow them, with strong bias toward local roads. The generator checks nearby road classes, then decides whether a lot should remain open, become a house, or occasionally become a park or tower in denser pockets.

This is important for the visual result: the city reads as neighborhoods, not as a uniform checkerboard.

### Core Browser Tuning Variables

The tuning panel exposes the main generation controls. These are the most important variables if you want to change density or road style.

#### Highway Controls

- `highwaySeeds` - Number of edge seeds used to start highway growth.
- `highwayMinLenRatio` - Minimum highway length relative to map size.
- `highwayMaxLenMult` - Maximum highway length multiplier.

#### Main Road Controls

- `mainBranchCap` - Maximum number of main-road branches.
- `mainFromHighwayFactor` - How aggressively main roads branch from highways.
- `mainSkipChance` - Chance to skip a potential main-road branch.

#### Local Road Controls

- `localBaseCap` - Base cap for total local-road growth.
- `localAreaFactor` - Local-road growth based on map area.
- `localFromMainFactor` - Local-road growth based on main-road length.
- `localBranchExtra` - Extra branching allowed from local roads.
- `localLoopChance` - Chance to form small local loops.

#### Housing and Cleanup Controls

- `houseSpawnMult` - Multiplier that boosts the final probability of placing buildings.
- `localFrontageBias` - Extra preference for lots facing local roads.
- `nearLocalBiasCap` - Upper limit on bonus density from nearby local roads.
- `mainFrontageCap` - Maximum density allowed on main-road frontage.
- `infillChance` - Chance to fill leftover gaps inside active local neighborhoods.
- `edgePruneLocal` - How aggressively local roads are removed near map edges.
- `edgePruneMain` - How aggressively main roads are removed near map edges.

### Browser UI Controls

- Speed buttons: `0.5x`, `1x`, and `3x`.
- Zoom controls: zoom in, zoom out, and reset view.
- Pause toggle.
- Closable generation tuning panel.
- Generation progress overlay during map creation.

## Python Simulation Stack

The Python side of the project is designed for simulation, debugging, and experimentation.

### `grid.py`

`CityGrid` stores the city as a 2D numpy array and provides helpers for:

- reading and writing cell types,
- checking bounds,
- finding drivable neighbors,
- generating preset grid cities,
- generating random stress-test layouts.

Cell types used by the Python stack:

- `EMPTY`
- `ROAD`
- `HIGHWAY`
- `INTERSECTION`

### `traffic_sim.py`

This module contains the actual traffic engine.

Main responsibilities:

- A* pathfinding over the road graph.
- Spawning cars at random drivable tiles.
- Advancing cars one tick at a time.
- Preventing collisions using an occupancy set.
- Limiting intersections to one entering car per tick.
- Tracking per-tick metrics.

#### SimulationEngine Parameters

- `max_cars` - Maximum number of active cars allowed at once.
- `spawn_rate` - Probability of spawning a new car each tick.

#### Car State

Each car tracks:

- `car_id` - unique identifier,
- `position` - current tile,
- `destination` - target tile,
- `path` - remaining route,
- `travel_time` - total ticks since spawn,
- `stopped` - whether the car could not move on the current tick,
- `waiting_at_intersection` - queue time spent blocked at intersections.

#### Metrics Returned by the Engine

- `tick` - current simulation tick.
- `active_cars` - number of cars currently in motion.
- `completed_cars` - number of cars that reached their destination.
- `avg_travel_time` - average travel time across active and completed cars.
- `stopped_cars` - number of cars that could not move on the latest tick.
- `congestion_map` - cars-per-tile array used for rendering and reward shaping.

### `environment.py`

This file wraps the simulation in a Gym-style interface suitable for reinforcement learning.

#### Observation Space

The observation is a 3-channel array with shape `(3, H, W)`:

- Channel 0 - normalized grid layout.
- Channel 1 - normalized congestion map.
- Channel 2 - binary intersection mask.

#### Action Space

Actions are encoded as a discrete integer:

- `0` - add road
- `1` - upgrade to highway
- `2` - add intersection

The tile index is packed into the same integer, so the environment can scale with grid size.

#### Reward Function

The reward is shaped to penalize inefficient or congested layouts:

- average travel time penalty,
- stopped car penalty,
- congestion penalty,
- build penalty when the action changes the grid.

This makes the environment useful for optimization experiments, even though it is not yet wired into a full RL training pipeline.

### `visualize.py`

This module provides a pygame renderer for the Python simulation.

It draws:

- the underlying city grid,
- congestion overlays,
- vehicles,
- a small HUD with current metrics.

If pygame is not installed, the script exits cleanly with an installation message.

### `main.py`

This is the command-line entry point.

Available run modes:

- Visual mode: opens the pygame renderer.
- Headless mode: runs the simulation without a display and prints metrics.
- Environment test mode: exercises the Gym-style wrapper with random actions.

## Installation

### Requirements

- Python 3.10 or newer recommended.
- `pygame` for the visual Python simulation.
- `numpy` for the grid and environment logic.

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you only want to use the browser version, you do not need the Python dependencies.

## Usage

### Browser Version

Open `city_visual.html` in a modern browser. The page generates a city automatically on load.

Use the on-screen controls to:

- change simulation speed,
- zoom and pan around the map,
- pause the traffic simulation,
- hide or show the tuning panel,
- regenerate the city after changing generation parameters.

### Python Visual Mode

```bash
python main.py
```

This launches the pygame version of the simulation.

### Python Headless Mode

```bash
python main.py --mode headless --ticks 500
```

This runs the simulation without a window and prints periodic metrics to the terminal.

### Environment Smoke Test

```bash
python main.py --mode envtest --ticks 50
```

This runs the Gym-style wrapper with random actions and prints a simple diagnostic summary.

## Simulation Parameters Worth Tuning

If you are using this project as a portfolio piece, these are the variables that matter most when describing the design.

### Road Structure

- Increase `highwaySeeds` for more citywide corridors.
- Increase `mainBranchCap` and `mainFromHighwayFactor` to create a larger collector network.
- Increase `localBaseCap`, `localAreaFactor`, and `localBranchExtra` to produce denser neighborhoods.
- Increase `localLoopChance` to create more organic block structure.

### Residential Density

- Increase `houseSpawnMult` to make empty lots more likely to fill.
- Increase `localFrontageBias` to push more housing onto local roads.
- Increase `nearLocalBiasCap` to let neighborhood density spread inward from local road frontage.
- Increase `infillChance` to reduce empty gaps inside dense districts.
- Lower `mainFrontageCap` if you want main-road parcels to stay sparse.

### Cleanup Behavior

- Lower `edgePruneLocal` and `edgePruneMain` if you want the city to keep more edge structure.
- Raise them if you want a more trimmed, centralized city footprint.

## Design Notes

A few implementation choices shape the look and behavior of the project:

- Roads are drawn from the rasterized grid rather than from raw path vertices, which keeps the visual output aligned with the actual simulation state.
- Local housing is best produced by explicit district stamping and local-road neighborhoods rather than by trying to brute-force road randomness.
- Intersections are treated as special junction tiles so routing can handle merges and crossing points cleanly.
- Cars occupy one tile at a time, which keeps collision logic simple and makes congestion readable.
- The project intentionally separates visual presentation from simulation logic so the browser and Python versions can evolve independently.

## Portfolio Angle

This project demonstrates more than procedural graphics. It combines:

- procedural generation,
- urban layout heuristics,
- traffic simulation,
- A* pathfinding,
- interactive visualization,
- parameter tuning,
- and a Gym-style environment for future ML work.

That combination makes it useful as a portfolio piece because it shows both systems thinking and visual polish.

## Known Limitations

- The browser and Python implementations are separate, not a single shared engine.
- The RL environment is a scaffold rather than a fully trained agent setup.
- City quality is heavily influenced by tuning, so generated layouts can vary a lot between runs.
- The project is optimized for readability and experimentation, not real-world urban accuracy.

## Suggested Next Steps

If you want to take the project further, the most natural upgrades are:

- add saved presets for different city styles,
- expose the browser tuning panel as named profiles,
- add better district zoning types,
- make the Python environment compatible with a full Gymnasium wrapper,
- collect benchmark metrics over repeated seeds,
- export screenshots or animated runs for portfolio presentation.

## License

No license has been specified yet.
