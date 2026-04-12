# AI City Planner

## Project Goal
Build an AI system that generates and improves city road layouts based on traffic performance.

## Core Idea

1. Generate a city road network.
2. Simulate traffic with cars (A* pathfinding, movement, congestion).
3. Measure performance (travel time, stopped cars, congestion).
4. Let an optimizer modify the road network.
5. Repeat and keep better layouts.

## Final Result
A simulation where an AI redesigns a city to reduce congestion, with before/after metrics and optional visualization.

## Current Implementation

## Latest Browser Release (V6)

The browser simulator has been upgraded to `V6` with a major systems pass.

- Rebuilt road generation pipeline (highways, arterials, locals) for more believable layouts.
- Randomized city boundary shape so each run has a more distinct footprint.
- Added in-app city editing tools plus `window.cityEditor` hooks for future AI-driven edits.
- Improved car recovery behavior with reroute cooldowns and deadlock escape logic.
- Expanded simulator controls/panels and updated export metadata to version `V6`.

### Browser simulation
- `city_visual.html`
- Procedural city generation + live traffic rendering and controls.

### Python optimization stack
- `grid.py`: 2D road-grid model and mutation helpers.
- `traffic_sim.py`: vehicle simulation engine + A* routing + congestion metrics.
- `environment.py`: Gym-style wrapper for RL experimentation.
- `visualize.py`: pygame renderer for baseline/optimized runs.
- `main.py`: CLI entrypoint for simulation and optimization modes.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Optimization loop (default mode):

```bash
python main.py --mode optimize
```

Useful optimization options:

```bash
python main.py --mode optimize --ticks 300 --iterations 25
python main.py --mode optimize --candidates-per-iter 12 --edits-per-candidate 5
python main.py --mode optimize --visualize-results
```

Other modes:

```bash
python main.py --mode visual
python main.py --mode headless
python main.py --mode envtest
```

Browser mode:

Open `city_visual.html` in a browser to use the interactive city generator/editor.

Latest release screenshot: `version-6.png`.

## Optimization Output

`optimize` mode reports:

- Baseline score
- Iteration-by-iteration accepted improvements
- Final before/after comparison for:
  - mean travel time
  - mean stopped cars
  - mean congestion
  - total optimization score

With `--visualize-results`, it also shows two pygame runs:

- Before Optimization
- After Optimization

## Notes

- The current optimizer is a hill-climbing AI that applies road edits and keeps better candidates.
- `environment.py` remains available for training a learned policy later (PPO/SB3 path).
