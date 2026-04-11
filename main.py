from grid import CityGrid
from traffic_sim import TrafficSimulation
from visualize import run_visualization


def main() -> None:
    grid = CityGrid(width=56, height=56)
    sim = TrafficSimulation(
        grid=grid,
        spawn_probability=0.62,
        max_cars=1600,
        seed=12,
    )
    run_visualization(grid, sim)


if __name__ == "__main__":
    main()
