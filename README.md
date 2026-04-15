# AI City Planner

AI City Planner is a traffic-simulation and layout-optimization project with two execution surfaces:

- A Python optimization stack for repeatable experiments.
- A browser simulator for interactive city generation, editing, and visualization.

The project objective is simple: generate road networks, simulate traffic, score performance, and iteratively improve layouts.

## Why This Project Exists

Most city demos stop at map generation. This project focuses on the full loop:

1. Build a drivable road graph.
2. Route vehicles with A*.
3. Measure flow and congestion over many ticks.
4. Apply edits to the network.
5. Keep changes only when system-level metrics improve.

That gives a practical baseline for future RL or search-based planning policies.

## Latest Updates (Apr 2026)

### Browser release V7

- Moved AI training fully out of the browser and into external Python training/runtime files.
- Added a local bridge service so Train AI, Stop AI, and Apply AI Layout call Python endpoints instead of in-page hill climbing.
- Added background training support and bridge status/log reporting for longer runs.
- Removed unused browser-side AI optimizer logic to reduce in-page overhead.
- Updated browser release metadata and map export version to V7.

### Browser release V6.5

- Added a bottom-left zoning toggle that draws colorful zone-boundary outlines.
- Added a zone legend panel that appears when zoning overlay is enabled.
- Improved district realism with more organic zone shape warping.
- Added more special buildings per zone (landmarks/civic, apartments/villas, plants) for clearer district identity.
- Updated browser release metadata and map export version to V6.5.

### Browser release V6.4

- Fixed the intersection lockup issue by removing over-restrictive junction entry gates.
- Kept movement safety while enabling smoother conflict resolution at intersection edges.
- Added demand-driven multi-spawn tick logic for more realistic active city traffic.
- Rebalanced density and finalized the active-car cap at 500 for stable runtime behavior.
- Updated browser release metadata and map export version to V6.4.

### Browser release V6.3

- Merged optimized rendering with offscreen caching for ground, buildings, roads, and minimap roads.
- Replaced linear-scan A* frontier selection with a binary-heap priority queue.
- Removed per-tick full car sorting in favor of a linear-time prioritized drive-order pass.
- Added precomputed signal-intersection cache to avoid full-map light updates each tick.
- Throttled HUD updates and refreshed release metadata/UI labels to V6.3.

### Browser release V6.2

- Stabilized highway spacing with stronger dominant-axis corridor separation.
- Reduced near-parallel shifted highway artifacts.
- Preserved sparse ramp spacing and highway-favoring travel costs.

## High-Level Architecture

```mermaid
flowchart LR
    A[Layout Generator] --> B[CityGrid]
    B --> C[SimulationEngine]
    C --> D[Metrics]
    D --> E[Optimizer]
    E -->|Mutations| B
    D --> F[Renderer / Reports]
```

## Repository Map

### Core Python modules

- `main.py`
  - CLI entrypoint.
  - Runs baseline, visual, optimization, and env test modes.
  - Contains hill-climbing optimization loop and score function.

- `grid.py`
  - `CityGrid` data model (numpy 2D grid).
  - Tile semantics: `EMPTY`, `ROAD`, `HIGHWAY`, `INTERSECTION`.
  - Includes generation and mutation helpers.

- `traffic_sim.py`
  - A* pathfinding with `heapq` priority queue.
  - `Car` entity model and per-tick update logic.
  - `SimulationEngine` for spawning, movement, collision checks, and metrics.

- `visualize.py`
  - Pygame renderer for simulation playback.
  - Draws tiles, cars, congestion overlays, and HUD metrics.

- `environment.py`
  - Gym-style wrapper for RL integration experiments.

### Browser simulator

- `city_visual.html`
  - Procedural city generation with hierarchical roads.
  - Real-time traffic simulation and controls.
  - In-app editor tools, minimap, metric panels, and map import/export.

### City generation theory

The browser generator is best understood as a constrained stochastic field on a 2D lattice. Each cell at integer coordinates $(x,y)$ is assigned a semantic state, but the layout is not random noise: it is built from layered spatial rules that bias the city toward a dense core, a structured road hierarchy, and a softer suburban edge.

The city boundary is modeled as a warped ellipse. With center $(c_x,c_y)$ and radii $(r_x,r_y)$, the normalized position is

$$
u = \frac{x-c_x}{r_x},\quad v = \frac{y-c_y}{r_y},\quad d = \sqrt{u^2+v^2}.
$$

The boundary mask is then given by

$$
M(x,y)=\mathbf{1}\{u^2+v^2 \le (1+w(\theta))^2\},\quad \theta=\operatorname{atan2}(v,u),
$$

where $w(\theta)$ is a low-amplitude angular perturbation. This keeps the city compact but avoids a perfectly circular or elliptical silhouette.

Zoning is radial at the top level and angularly warped at the finer level. In practice, the generator applies thresholds on $d$ to produce a CBD core, a commercial ring, and a residential belt, then introduces secondary commercial clusters and industrial edge bias. Conceptually, the rule is:

$$
z(x,y)=
\begin{cases}
\operatorname{CBD}, & d < d_1 + \delta(\theta) \\
\operatorname{Commercial}, & d_1 \le d < d_2 \\
\operatorname{Residential}, & d \ge d_2
\end{cases}
$$

with small perturbations $\delta(\theta)$ so district borders are not concentric rings.

The road network is hierarchical rather than uniform. Highways are long edge-to-edge corridors that enforce global connectivity; arterials form a more regular collector grid with spacing constraints; local streets emerge as constrained growth from arterials. In graph terms, the generator first creates a sparse backbone $G_H$, then overlays a denser intermediate graph $G_A$, and finally fills the remaining reachable space with local branches $G_L$, subject to cleanup rules that remove dead ends and isolated segments.

This structure matters mathematically because it balances three competing objectives:

1. Connectivity, so the drivable subgraph stays largely reachable.
2. Separation, so parallel highways and arterials do not collapse into clumps.
3. Variety, so the city does not reduce to a perfect grid or a pure random field.

In short, the generator behaves like a multi-scale spatial process: a warped domain, a radial zoning prior, and a connectivity-preserving road growth model.

## Python Runtime Flow

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant Grid as CityGrid
    participant Sim as SimulationEngine
  participant Search as HillClimber

    CLI->>Grid: make_city(layout)
    CLI->>Sim: initialize(grid)
    loop ticks
        Sim->>Sim: step()
        Sim-->>CLI: metrics snapshot
    end
    CLI->>Search: evaluate and mutate candidates
    Search->>Grid: apply_random_edit()
    Search-->>CLI: improved grid/score
```

## Data Model Summary

### Grid state

- Representation: `numpy.ndarray` shaped `(height, width)`.
- Coordinates: `(x, y)` but stored as `grid[y, x]`.
- Drivable cells: road, highway, intersection.

### Vehicle state

- Current tile position.
- Destination tile.
- Remaining waypoint path.
- Travel-time accumulator.
- Stopped flag and intersection wait behavior.

### Metrics tracked

- `avg_travel_time`
- `stopped_cars`
- `congestion_map`
- `completed_cars`
- `active_cars`

## Optimization Strategy (Current)

The current optimizer is a hill-climbing search in `main.py`:

1. Evaluate baseline layout.
2. Clone current best layout.
3. Apply random edits to candidate layouts.
4. Simulate each candidate for fixed ticks.
5. Accept only strict score improvements.

Current objective score (lower is better):

$$
  ext{score} = 1.0\cdot\overline{travel} + 2.0\cdot\overline{stopped} + 40.0\cdot\overline{congestion}
$$

## CLI Usage

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run optimization (default workflow)

```bash
python main.py --mode optimize
```

### 3) Useful optimization variants

```bash
python main.py --mode optimize --ticks 300 --iterations 25
python main.py --mode optimize --candidates-per-iter 12 --edits-per-candidate 5
python main.py --mode optimize --visualize-results
```

### 4) Other modes

```bash
python main.py --mode visual
python main.py --mode headless
python main.py --mode envtest
python main.py --mode rltrain --episodes 500 --episode-length 250
```

## External RL Training (Headless)

The browser file is no longer required for learning. Training now runs in Python via `rl_trainer.py` and can run for long sessions in the background.

### Quick start

```bash
python main.py --mode rltrain --episodes 600 --episode-length 250 --eval-every 25
```

Model artifacts are saved to:

- `models/linear_double_q_weights.npz`
- `models/linear_double_q_weights.json`

### RL math used

The trainer uses **Linear Double Q-learning**:

$$
Q(s,a) = w^\top \phi(s,a)
$$

with Double-Q target:

$$
y = r + \gamma Q_{\bar{i}}(s', \arg\max_{a'} Q_i(s',a'))
$$

and update:

$$
w_i \leftarrow w_i + \alpha (y - Q_i(s,a))\phi(s,a)
$$

The environment reward in `environment.py` is delta-based:

$$
r_t =
w_1(\overline{T}_{t-1}-\overline{T}_t) +
w_2(S_{t-1}-S_t) +
w_3(C_{t-1}-C_t) +
w_4(\Delta \text{completed}) - \text{buildPenalty} - \text{invalidPenalty}
$$

where lower travel/stops/congestion gives positive reward.

## Browser Simulator Notes

Open `city_visual.html` directly in your browser.

It includes:

- City generation + simulation controls.
- Zoom/pan/minimap.
- Interactive road/zoning editor.
- Stop-reason and congestion diagnostics.
- Import/export of map state.

Version history is tracked in `portfolio_change_log.txt`.

### Train AI button bridge (external Python)

The browser no longer runs hill-climbing AI inside HTML. The **Train AI / Stop AI / Apply AI Layout** buttons now call a local Python bridge server.

Start the bridge in a terminal before using those buttons:

```bash
python bridge_ai_server.py --host 127.0.0.1 --port 8765
```

How it works:

- `Train AI` starts external RL training (`main.py --mode rltrain`) in the background.
- `Stop AI` sends a terminate signal to the running trainer.
- `Apply AI Layout` sends the current browser map state to Python, applies the trained policy, and restores the returned state in the browser.

Training log path:

- `models/bridge_train.log`

## Extension Points

If you want to evolve this codebase, these are the best insertion points:

- Replace hill climbing with learned policy search in `main.py`.
- Add richer network edits in `grid.py` (lane count, one-way rules, turn penalties).
- Introduce signal policies and dynamic timing in `traffic_sim.py`.
- Feed richer observations/rewards through `environment.py`.
- Add reproducible experiment logging (CSV/JSON) for benchmark comparisons.

## Known Constraints

- Current optimizer is stochastic and local (can get stuck in local minima).
- Grid abstractions are intentionally coarse and lane-level realism is simplified.
- Browser and Python simulations are related but not identical implementations.

## Project Status

The project is currently a strong prototype baseline:

- Deterministic enough for repeatable scoring loops.
- Fast enough for iterative optimization experiments.
- Structured enough to serve as a foundation for RL-based planners.
