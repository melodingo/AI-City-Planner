# AI City Planner

Procedural city generation and traffic simulation, built to explore how road hierarchy, frontage rules, and congestion interact in a believable urban layout.

The repository contains two connected systems:

- `city_visual.html` - a browser-based city generator with live traffic, tuning controls, and a highly visual rendering pass.
- The Python stack - a grid model, traffic engine, renderer, and Gym-style environment for simulation and experimentation.

The main design goal is simple: make the city feel like a city. Roads should create districts, houses should gather where roads support them, and traffic should reveal whether the layout actually works.

## Quick Links

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Browser Simulation](#browser-simulation)
- [Python Simulation Stack](#python-simulation-stack)
- [Key Variables](#key-variables)
- [Installation](#installation)
- [Usage](#usage)
- [Project Notes](#project-notes)

## Overview

This project is built around a layered urban model:

- highways establish the long-range structure,
- main roads distribute flow into the city,
- local roads form dense residential networks,
- buildings cluster along frontage instead of filling the map uniformly,
- cars move through the network using pathfinding and collision-aware movement.

The browser version goes further by using explicit local-road district stamping and dense housing backfill so neighborhoods read as compact blocks instead of isolated tiles.

## What Makes It Interesting

- It is not just a road generator. It is a road-and-development system.
- It is not just a traffic sim. It measures whether the road plan actually functions.
- It uses the same core city logic in two forms: a visual HTML canvas version and a Python simulation stack.
- It exposes tunable variables so different city styles can be created without rewriting the generator.

## How It Works

The project separates generation into stages:

1. Build a city boundary mask.
2. Seed highways from the map edges.
3. Branch main roads from the highway network.
4. Grow local roads from the main network and from local neighborhoods.
5. Place buildings using frontage rules and local-density bias.
6. Spawn cars, route them through the network, and track traffic metrics.

That structure is what makes the output feel layered rather than random.

## Browser Simulation

The browser version in `city_visual.html` is the most complete presentation of the project. It renders the city on a canvas and includes a tuning panel so the layout can be adjusted in real time.

### Road Classes

The browser generator uses four road states:

- `LOCAL` - neighborhood roads and residential loops.
- `MAIN` - distributor roads that connect districts.
- `HIGHWAY` - long-distance arterial routes.
- `INTER` - intersections and junctions.

This hierarchy matters because it controls both the city shape and how buildings are allowed to appear.

### Building Logic

Buildings are placed only where road frontage makes sense. Local roads are strongly preferred, main-road frontage is capped more aggressively, and dense local pockets receive extra infill.

That means the city can produce patterns like:

- local road, house, house, local road,
- residential blocks with occasional gaps,
- denser clusters near active local networks,
- rare parks or taller structures inside crowded areas.

### UI Controls

The browser interface includes:

- simulation speed buttons,
- zoom in, zoom out, and reset view,
- panning by dragging,
- pause and resume,
- a closable generation tuning panel,
- a generation progress overlay while the map is being built.

## Key Variables

These are the knobs that matter most if you want to explain or evolve the generator.

| Variable | Meaning | Effect |
| --- | --- | --- |
| `highwaySeeds` | Number of highway entry points | Higher values create more regional structure |
| `highwayMinLenRatio` | Minimum highway length relative to map size | Higher values force longer trunk lines |
| `highwayMaxLenMult` | Maximum highway growth multiplier | Higher values let highways stretch further |
| `mainBranchCap` | Cap on main-road branching | Higher values create a larger collector network |
| `mainFromHighwayFactor` | Branch density from highways | Higher values push more main roads outward |
| `mainSkipChance` | Chance to skip a main-road branch | Lower values make main roads more continuous |
| `localBaseCap` | Base local-road budget | Higher values create more neighborhood roads |
| `localAreaFactor` | Local-road budget tied to map area | Higher values scale local density up globally |
| `localFromMainFactor` | Local-road budget tied to main roads | Higher values create stronger district growth |
| `localBranchExtra` | Extra branching from local roads | Higher values make districts more tangled and complete |
| `localLoopChance` | Chance to create local loops | Higher values make neighborhoods feel less grid-like |
| `houseSpawnMult` | Housing placement multiplier | Higher values increase fill across valid lots |
| `localFrontageBias` | Preference for local-road frontage | Higher values push more housing onto local streets |
| `nearLocalBiasCap` | Cap on nearby-local density bonus | Higher values spread density deeper into districts |
| `mainFrontageCap` | Ceiling for main-road frontage density | Lower values keep main streets less residential |
| `infillChance` | Chance to fill remaining local gaps | Higher values reduce holes inside neighborhoods |
| `edgePruneLocal` | Edge cleanup for local roads | Higher values trim local roads near borders more aggressively |
| `edgePruneMain` | Edge cleanup for main roads | Higher values trim main roads near borders more aggressively |

## Python Simulation Stack

The Python side is focused on clean simulation logic, testing, and future ML experimentation.

### `grid.py`

`CityGrid` stores the city as a 2D numpy array and provides helpers for:

- reading and writing cell types,
- checking bounds,
- finding drivable neighbors,
- creating preset grid-style cities,
- generating random stress-test layouts.

Cell types used by the Python stack:

- `EMPTY`
- `ROAD`
- `HIGHWAY`
- `INTERSECTION`

### `traffic_sim.py`

This is the core simulation engine.

It handles:

- A* pathfinding,
- spawning cars on drivable tiles,
- advancing cars one tick at a time,
- collision prevention through occupancy tracking,
- one-car-per-intersection-per-tick behavior,
- metrics collection for rendering and analysis.

#### Important Parameters

- `max_cars` - maximum number of active cars allowed at once.
- `spawn_rate` - probability of spawning a new car each tick.

#### Car State

Each car tracks:

- `car_id` - unique identifier,
- `position` - current tile,
- `destination` - target tile,
- `path` - remaining route,
- `travel_time` - total ticks since spawn,
- `stopped` - whether movement was blocked on the current tick,
- `waiting_at_intersection` - queue time spent waiting at junctions.

#### Metrics Returned by the Engine

- `tick` - current simulation tick.
- `active_cars` - number of cars currently moving.
- `completed_cars` - number of cars that reached their destination.
- `avg_travel_time` - average travel time across active and completed cars.
- `stopped_cars` - number of cars that could not move on the latest tick.
- `congestion_map` - cars-per-tile array used for rendering and reward shaping.

### `environment.py`

This file wraps the simulation in a Gym-style environment.

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

The tile index is packed into the same integer, which keeps the interface scalable with grid size.

#### Reward Function

The reward is shaped to penalize inefficient or congested layouts:

- average travel time penalty,
- stopped-car penalty,
- congestion penalty,
- build penalty when the action changes the grid.

This makes the environment useful as a prototype for optimization or RL experiments.

### `visualize.py`

This module provides the pygame renderer for the Python simulation.

It draws:

- the underlying city grid,
- congestion overlays,
- vehicles,
- a HUD with current metrics.

### `main.py`

This is the command-line entry point.

Available run modes:

- Visual mode - opens the pygame renderer.
- Headless mode - runs the simulation without a display and prints metrics.
- Environment test mode - exercises the Gym-style wrapper with random actions.

## Installation

### Requirements

- Python 3.10 or newer recommended.
- `numpy` for the grid and environment logic.
- `pygame` for the visual Python simulation.

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you only want the browser version, the Python dependencies are optional.

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

### Python Headless Mode

```bash
python main.py --mode headless --ticks 500
```

### Environment Smoke Test

```bash
python main.py --mode envtest --ticks 50
```

## Project Notes

### Why the Browser Version Looks Different

The browser generator is tuned to create stronger neighborhood density than a simple random fill approach. The important part is not just road probability. It is the interaction between:

- explicit local-road districts,
- frontage-aware housing placement,
- dense infill inside active neighborhoods,
- restrained main-road housing,
- and cleanup rules that preserve the district structure.

### Why the Python Version Still Matters

The Python stack is a cleaner simulation core. It is useful when you want:

- reproducible test runs,
- terminal-based debugging,
- a pygame rendering path,
- or a stepping stone toward reinforcement learning.

### Design Tradeoffs

- Roads are drawn from the rasterized grid rather than raw path vertices, which keeps the visualization aligned with the simulation state.
- Cars occupy one tile at a time, which keeps collision logic simple and readable.
- The browser and Python systems are separate, which keeps each implementation focused on its own strengths.
- The project favors legibility and urban feel over strict real-world accuracy.

## Portfolio Angle

This project demonstrates procedural generation, urban layout heuristics, simulation, pathfinding, interactive visualization, and tunable system design in a single package.

It is a strong portfolio piece because it shows that the code is not only functional, but also designed around a clear visual and behavioral goal.

## Limitations

- The browser and Python implementations are not a single shared engine.
- The RL environment is a scaffold, not a finished training pipeline.
- City quality still depends on generation settings and random seed.
- The project is optimized for experimentation and presentation, not geographic realism.

## Next Steps

Possible upgrades from here:

- add saved presets for different city styles,
- expose named generation profiles,
- add explicit district types beyond simple residential density,
- build a full Gymnasium wrapper,
- collect benchmark metrics over repeated seeds,
- export screenshots or short clips for portfolio use.

## License

No license has been specified yet.
