"""Smoke tests for the optimizer corpus, motion models, and pathfinders."""

from __future__ import annotations

import math

import pytest

from nio.base import sphere
from nio.motion import boids_step, vicsek_step
from nio.pathfinding import AntColonyPath, OccupancyGrid, RiverFormationDynamics
from nio.registry import OPTIMIZERS


def _in_bounds(position, bounds):
    return all(lo - 1e-6 <= value <= hi + 1e-6 for value, (lo, hi) in zip(position, bounds))


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_optimizer_runs_on_sphere(name):
    cls = OPTIMIZERS[name]
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    optimizer = cls(objective=sphere, bounds=bounds, population_size=12, seed=1)
    position, value = optimizer.run(8)
    assert len(position) == 2
    assert math.isfinite(value)
    assert _in_bounds(position, bounds)


@pytest.mark.parametrize(
    "name",
    ["pso", "de", "cs", "gwo"],
)
def test_canonical_optimizers_improve_on_sphere(name):
    cls = OPTIMIZERS[name]
    bounds = [(-5.0, 5.0)] * 5
    probe = cls(objective=sphere, bounds=bounds, population_size=24, seed=2)
    probe.initialise()
    initial = probe.best_fitness
    position, value = cls(objective=sphere, bounds=bounds, population_size=24, seed=2).run(80)
    assert math.isfinite(value)
    assert value < initial * 0.25
    assert _in_bounds(position, bounds)


def test_boids_and_vicsek_keep_group_size():
    positions = [[0.0, 0.0], [1.0, 0.2], [0.2, 1.0]]
    velocities = [[0.1, 0.0], [0.0, 0.1], [-0.1, 0.05]]
    moved, speeds = boids_step(positions, velocities, anchors=[[4.0, 0.0]], anchor_weight=0.2)
    assert len(moved) == 3
    assert len(speeds) == 3
    assert all(math.isfinite(coord) for point in moved for coord in point)

    headings = [0.0, 0.4, -0.2]
    stepped, new_headings = vicsek_step(positions, headings, radius=2.0, noise=0.05, speed=0.5)
    assert len(stepped) == 3
    assert len(new_headings) == 3


def test_ant_colony_and_river_formation_cross_a_gap():
    blocked = {(row, 2) for row in range(4)}
    grid = OccupancyGrid(5, 5, blocked)
    start, goal = (0, 0), (0, 4)
    ants = AntColonyPath(grid, start, goal, n_ants=12, iterations=15, seed=1).run()
    river = RiverFormationDynamics(grid, start, goal, n_drops=16, iterations=20, seed=1).run()
    for path in (ants, river):
        assert path[0] == start
        assert path[-1] == goal
        assert all(cell not in blocked for cell in path)
