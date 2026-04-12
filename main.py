"""
main.py
-------
Entry point for AI City Planner.

Project loop:
1) Generate a city road layout.
2) Simulate traffic.
3) Measure congestion/travel metrics.
4) Let an optimizer modify roads.
5) Repeat and keep improvements.

Run examples
------------
python main.py --mode optimize --ticks 300 --iterations 20
python main.py --mode optimize --visualize-results
python main.py --mode visual
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, Tuple

import numpy as np

from grid import CityGrid
from traffic_sim import SimulationEngine


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def make_city(width: int, height: int, layout: str) -> CityGrid:
    """Create a fresh city layout using one of the available generators."""
    grid = CityGrid(width=width, height=height)
    if layout == "random":
        grid.build_random_city(road_density=0.30)
    else:
        grid.build_grid_city(block_size=4)
    return grid


def clone_grid(src: CityGrid) -> CityGrid:
    """Deep-copy a CityGrid so candidates can be evaluated independently."""
    dst = CityGrid(width=src.width, height=src.height)
    dst.grid = src.grid.copy()
    return dst


def evaluate_city(
    grid: CityGrid,
    ticks: int,
    max_cars: int,
    spawn_rate: float,
    seed: int,
) -> Dict[str, float]:
    """Run one simulation rollout and aggregate objective metrics."""
    state_before = random.getstate()
    random.seed(seed)

    engine = SimulationEngine(grid, max_cars=max_cars, spawn_rate=spawn_rate)

    travel_vals = []
    stopped_vals = []
    congestion_vals = []

    for _ in range(ticks):
        m = engine.step()
        travel_vals.append(float(m["avg_travel_time"]))
        stopped_vals.append(float(m["stopped_cars"]))
        congestion_vals.append(float(np.mean(m["congestion_map"])))

    final = engine.metrics()
    random.setstate(state_before)

    mean_travel = float(np.mean(travel_vals)) if travel_vals else 0.0
    mean_stopped = float(np.mean(stopped_vals)) if stopped_vals else 0.0
    mean_congestion = float(np.mean(congestion_vals)) if congestion_vals else 0.0

    # Lower score is better.
    score = (
        1.0 * mean_travel
        + 2.0 * mean_stopped
        + 40.0 * mean_congestion
    )

    return {
        "score": score,
        "mean_travel_time": mean_travel,
        "mean_stopped_cars": mean_stopped,
        "mean_congestion": mean_congestion,
        "completed_cars": float(final["completed_cars"]),
        "active_cars": float(final["active_cars"]),
    }


def apply_random_edit(grid: CityGrid, rng: random.Random) -> bool:
    """Try one random network edit. Returns True if the grid changed."""
    x = rng.randrange(grid.width)
    y = rng.randrange(grid.height)
    action = rng.choice(("add_road", "upgrade_highway", "add_intersection"))

    if action == "add_road":
        return grid.add_road(x, y)
    if action == "upgrade_highway":
        return grid.upgrade_to_highway(x, y)
    return grid.add_intersection(x, y)


def optimize_city(
    base_grid: CityGrid,
    ticks: int,
    max_cars: int,
    spawn_rate: float,
    iterations: int,
    candidates_per_iter: int,
    edits_per_candidate: int,
    seed: int,
) -> Tuple[CityGrid, Dict[str, float], Dict[str, float], int]:
    """
    Simple hill-climbing optimizer:
    - start from baseline grid
    - sample edited candidates
    - keep candidate if score improves
    """
    rng = random.Random(seed)

    baseline_grid = clone_grid(base_grid)
    best_grid = clone_grid(base_grid)
    baseline_metrics = evaluate_city(
        baseline_grid, ticks=ticks, max_cars=max_cars, spawn_rate=spawn_rate, seed=seed
    )
    best_metrics = dict(baseline_metrics)

    accepted = 0
    print("=== Optimization Loop ===")
    print(f"Baseline score: {baseline_metrics['score']:.3f}")

    for it in range(1, iterations + 1):
        round_best_grid = None
        round_best_metrics = None

        for c in range(candidates_per_iter):
            cand = clone_grid(best_grid)
            changed = 0
            for _ in range(edits_per_candidate):
                if apply_random_edit(cand, rng):
                    changed += 1

            if changed == 0:
                continue

            cand_metrics = evaluate_city(
                cand,
                ticks=ticks,
                max_cars=max_cars,
                spawn_rate=spawn_rate,
                seed=seed + it * 1000 + c,
            )

            if cand_metrics["score"] < best_metrics["score"]:
                if round_best_metrics is None or cand_metrics["score"] < round_best_metrics["score"]:
                    round_best_grid = cand
                    round_best_metrics = cand_metrics

        if round_best_grid is not None and round_best_metrics is not None:
            best_grid = round_best_grid
            best_metrics = round_best_metrics
            accepted += 1
            print(
                f"Iter {it:>3}: improved score -> {best_metrics['score']:.3f} "
                f"(travel {best_metrics['mean_travel_time']:.2f}, "
                f"stopped {best_metrics['mean_stopped_cars']:.2f})"
            )
        else:
            print(f"Iter {it:>3}: no improvement")

    return best_grid, baseline_metrics, best_metrics, accepted


# ---------------------------------------------------------------------------
# Existing simulation modes
# ---------------------------------------------------------------------------

def run_headless(ticks: int = 200) -> None:
    """Run one baseline simulation without a display and print metrics."""
    print("=== City Traffic Simulation - Headless Mode ===\n")

    grid = make_city(width=20, height=20, layout="grid")
    engine = SimulationEngine(grid, max_cars=30, spawn_rate=0.3)

    print(f"{'Tick':>5} | {'Cars':>4} | {'Done':>4} | {'AvgTime':>8} | {'Stopped':>7}")
    print("-" * 42)

    for t in range(1, ticks + 1):
        m = engine.step()
        if t % 20 == 0 or t == 1:
            print(
                f"{m['tick']:>5} | "
                f"{m['active_cars']:>4} | "
                f"{m['completed_cars']:>4} | "
                f"{m['avg_travel_time']:>8.2f} | "
                f"{m['stopped_cars']:>7}"
            )


def run_visual_from_grid(grid: CityGrid, ticks: int, title: str) -> None:
    """Render a specific grid in pygame for a fixed number of ticks."""
    try:
        from visualize import CityRenderer
    except SystemExit:
        print("Falling back to headless mode (pygame not available).")
        run_headless(ticks)
        return

    engine = SimulationEngine(grid, max_cars=30, spawn_rate=0.3)
    renderer = CityRenderer(grid, cell_size=32, fps=10, show_congestion=True, title=title)

    for _ in range(5):
        engine.step()

    t = 0
    running = True
    while running and t < ticks:
        metrics = engine.step()
        running = renderer.draw(engine.cars, metrics, congestion_map=metrics["congestion_map"])
        t += 1

    renderer.close()


def run_visual(ticks: int = 500) -> None:
    """Run baseline simulation in pygame."""
    grid = make_city(width=20, height=20, layout="grid")
    run_visual_from_grid(grid, ticks=ticks, title="City Traffic Simulation - Baseline")


def run_env_test(ticks: int = 50) -> None:
    """Quick sanity-check of the Gym-style environment wrapper."""
    from environment import TrafficEnv

    print("=== RL Environment Smoke Test ===\n")
    env = TrafficEnv(width=20, height=20, episode_length=ticks)
    obs = env.reset()
    print(f"Observation shape : {obs.shape}")
    print(f"Action space size : {env.action_space_n}\n")

    total_reward = 0.0
    for step in range(ticks):
        action = random.randrange(env.action_space_n)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(
                f"Step {step:>3} | reward {reward:>7.2f} | "
                f"active {info['active_cars']:>3} | "
                f"done {done}"
            )
        if done:
            break

    print(f"\nTotal reward over episode: {total_reward:.2f}")


# ---------------------------------------------------------------------------
# Goal-aligned optimization mode
# ---------------------------------------------------------------------------

def run_optimize(
    ticks: int,
    iterations: int,
    candidates_per_iter: int,
    edits_per_candidate: int,
    seed: int,
    visualize_results: bool,
) -> None:
    """Run baseline-vs-optimized city comparison."""
    base_grid = make_city(width=20, height=20, layout="grid")

    best_grid, baseline, improved, accepted = optimize_city(
        base_grid,
        ticks=ticks,
        max_cars=30,
        spawn_rate=0.3,
        iterations=iterations,
        candidates_per_iter=candidates_per_iter,
        edits_per_candidate=edits_per_candidate,
        seed=seed,
    )

    def pct_delta(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100.0

    print("\n=== Before vs After ===")
    print(f"Accepted improvements: {accepted}/{iterations}")
    print(f"Score            : {baseline['score']:.3f} -> {improved['score']:.3f} ({pct_delta(baseline['score'], improved['score']):+.1f}%)")
    print(f"Mean travel time : {baseline['mean_travel_time']:.3f} -> {improved['mean_travel_time']:.3f} ({pct_delta(baseline['mean_travel_time'], improved['mean_travel_time']):+.1f}%)")
    print(f"Mean stopped cars: {baseline['mean_stopped_cars']:.3f} -> {improved['mean_stopped_cars']:.3f} ({pct_delta(baseline['mean_stopped_cars'], improved['mean_stopped_cars']):+.1f}%)")
    print(f"Mean congestion  : {baseline['mean_congestion']:.5f} -> {improved['mean_congestion']:.5f} ({pct_delta(baseline['mean_congestion'], improved['mean_congestion']):+.1f}%)")

    if visualize_results:
        print("\nShowing baseline city first. Close window (or press ESC) to continue.")
        run_visual_from_grid(clone_grid(base_grid), ticks=min(ticks, 400), title="Before Optimization")
        print("Showing optimized city. Close window (or press ESC) to finish.")
        run_visual_from_grid(clone_grid(best_grid), ticks=min(ticks, 400), title="After Optimization")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI City Planner")
    parser.add_argument(
        "--mode",
        choices=["visual", "headless", "envtest", "optimize"],
        default="optimize",
        help="visual: pygame baseline | headless: text metrics | envtest: RL wrapper smoke test | optimize: AI improvement loop",
    )
    parser.add_argument("--ticks", type=int, default=300, help="Ticks per evaluation/simulation run")
    parser.add_argument("--iterations", type=int, default=20, help="Optimization iterations")
    parser.add_argument("--candidates-per-iter", type=int, default=10, help="Candidate layouts sampled each iteration")
    parser.add_argument("--edits-per-candidate", type=int, default=4, help="Road edits applied per candidate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible optimization")
    parser.add_argument(
        "--visualize-results",
        action="store_true",
        help="In optimize mode, show pygame before/after visualization",
    )
    args = parser.parse_args()

    if args.mode == "headless":
        run_headless(args.ticks)
    elif args.mode == "envtest":
        run_env_test(args.ticks)
    elif args.mode == "visual":
        run_visual(args.ticks)
    else:
        run_optimize(
            ticks=args.ticks,
            iterations=args.iterations,
            candidates_per_iter=args.candidates_per_iter,
            edits_per_candidate=args.edits_per_candidate,
            seed=args.seed,
            visualize_results=args.visualize_results,
        )


if __name__ == "__main__":
    main()
