"""Leaders agree on a waypoint by minimising a short-horizon path cost.

Any registered continuous optimizer can propose the waypoint. Ant colony and
river formation dynamics can be asked for a grid path instead.
"""

from __future__ import annotations

from typing import List, Sequence

from nio.pathfinding import AntColonyPath, RiverFormationDynamics
from nio.registry import get_optimizer

from .env import LeaderSwarmEnv


def plan_actions(
    env: LeaderSwarmEnv,
    algorithm: str = "gwo",
    iterations: int = 18,
    population_size: int = 14,
) -> List[int]:
    """Choose one shared waypoint, then aim every leader at it."""
    origin = env.leader_centroid()

    def cost(position: Sequence[float]) -> float:
        return env.waypoint_cost(position, origin)

    optimizer = get_optimizer(algorithm)(
        objective=cost,
        bounds=[(0.0, env.width), (0.0, env.height)],
        population_size=population_size,
        seed=env.steps + 17,
    )
    waypoint, _value = optimizer.run(iterations)
    return [env.action_toward(index, waypoint) for index in range(env.n_leaders)]


def plan_actions_grid(env: LeaderSwarmEnv, method: str = "aco", resolution: int = 16) -> List[int]:
    """Rasterise the world and follow an ant-colony or river-formation path."""
    grid, start, goal = env.occupancy(resolution)
    seed = env.steps + 3
    if method == "rfd":
        path = RiverFormationDynamics(grid, start, goal, seed=seed).run()
    elif method == "aco":
        path = AntColonyPath(grid, start, goal, seed=seed).run()
    else:
        raise ValueError("method must be 'aco' or 'rfd'")
    if len(path) < 2 or path[-1] != goal:
        return plan_actions(env)
    lookahead = path[min(2, len(path) - 1)]
    waypoint = grid.cell_center(lookahead, env.width, env.height)
    return [env.action_toward(index, waypoint) for index in range(env.n_leaders)]


def fixed_heading_actions(env: LeaderSwarmEnv, heading: int = 0, fast: bool = True) -> List[int]:
    """Every leader flies the same compass point. East is heading 0."""
    speed_index = 1 if fast else 0
    action = speed_index * 8 + (heading % 8)
    return [action for _ in range(env.n_leaders)]
