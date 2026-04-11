"""
main.py
-------
Entry point for the city traffic simulation.

Run modes
---------
  python main.py             → pygame visualisation (default)
  python main.py --headless  → headless benchmark, prints metrics to stdout
  python main.py --ticks N   → run for N ticks (default 500)
"""

import argparse
import sys

from grid import CityGrid
from traffic_sim import SimulationEngine


# ---------------------------------------------------------------------------
# Headless run (no pygame needed)
# ---------------------------------------------------------------------------

def run_headless(ticks: int = 200) -> None:
    """Run the simulation without a display and print a metrics summary."""
    print("=== City Traffic Simulation – Headless Mode ===\n")

    grid = CityGrid(width=20, height=20)
    grid.build_grid_city(block_size=4)

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

    print("\nFinal grid:\n")
    print(grid)
    print("\nDone.")


# ---------------------------------------------------------------------------
# Visual run
# ---------------------------------------------------------------------------

def run_visual(ticks: int = 500) -> None:
    """Run the simulation with a pygame window."""
    try:
        from visualize import CityRenderer
    except SystemExit:
        print("Falling back to headless mode (pygame not available).")
        run_headless(ticks)
        return

    grid = CityGrid(width=20, height=20)
    grid.build_grid_city(block_size=4)

    engine   = SimulationEngine(grid, max_cars=30, spawn_rate=0.3)
    renderer = CityRenderer(grid, cell_size=32, fps=10, show_congestion=True)

    # Warm-up: populate some cars before opening the window
    for _ in range(5):
        engine.step()

    t = 0
    running = True
    while running and t < ticks:
        metrics = engine.step()
        cmap    = metrics["congestion_map"]

        running = renderer.draw(engine.cars, metrics, congestion_map=cmap)
        t += 1

    renderer.close()
    print(f"\nSimulation ended at tick {t}.")
    m = engine.metrics()
    print(f"  Active cars    : {m['active_cars']}")
    print(f"  Completed cars : {m['completed_cars']}")
    print(f"  Avg travel time: {m['avg_travel_time']:.2f} ticks")
    print(f"  Stopped cars   : {m['stopped_cars']}")


# ---------------------------------------------------------------------------
# RL environment smoke-test (demonstrates how to use environment.py)
# ---------------------------------------------------------------------------

def run_env_test(ticks: int = 50) -> None:
    """Quick sanity-check of the Gym-style environment."""
    from environment import TrafficEnv

    print("=== RL Environment Smoke Test ===\n")
    env = TrafficEnv(width=20, height=20, episode_length=ticks)
    obs = env.reset()
    print(f"Observation shape : {obs.shape}")
    print(f"Action space size : {env.action_space_n}\n")

    total_reward = 0.0
    for step in range(ticks):
        # Random action (replace with agent.predict(obs) for real RL)
        import random
        action = random.randrange(env.action_space_n)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f"Step {step:>3} | reward {reward:>7.2f} | "
                  f"active {info['active_cars']:>3} | "
                  f"done {done}")
        if done:
            break

    print(f"\nTotal reward over episode: {total_reward:.2f}")
    print("\nGrid snapshot:")
    print(env.render_text())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="City Traffic Simulation")
    parser.add_argument(
        "--mode",
        choices=["visual", "headless", "envtest"],
        default="visual",
        help="visual: pygame window | headless: text metrics | envtest: RL env test",
    )
    parser.add_argument(
        "--ticks", type=int, default=500,
        help="Number of simulation ticks to run (default: 500)",
    )
    args = parser.parse_args()

    if args.mode == "headless":
        run_headless(args.ticks)
    elif args.mode == "envtest":
        run_env_test(args.ticks)
    else:
        run_visual(args.ticks)


if __name__ == "__main__":
    main()
