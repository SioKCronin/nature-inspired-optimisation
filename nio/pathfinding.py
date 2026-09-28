"""Grid pathfinders for lead-drone routing.

Ant colony optimization and river formation dynamics both return a list of
cells from start to goal. Continuous ACO lives in :mod:`nio.acor`.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Cell = Tuple[int, int]


class OccupancyGrid:
    """Rectangular grid with blocked cells."""

    def __init__(self, rows: int, cols: int, blocked: Optional[Iterable[Cell]] = None) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("grid dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self.blocked: Set[Cell] = set(blocked or [])

    def contains(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols and cell not in self.blocked

    def neighbors(self, cell: Cell) -> List[Cell]:
        row, col = cell
        found = []
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + d_row, col + d_col)
            if self.contains(nxt):
                found.append(nxt)
        return found

    @classmethod
    def from_obstacles(
        cls,
        width: float,
        height: float,
        obstacles: Sequence[Sequence[float]],
        resolution: int = 16,
    ) -> "OccupancyGrid":
        """Rasterise circular obstacles ``(cx, cy, radius)`` into a grid."""
        blocked = []
        for row in range(resolution):
            for col in range(resolution):
                x = (col + 0.5) / resolution * width
                y = (row + 0.5) / resolution * height
                for obstacle in obstacles:
                    cx, cy, radius = obstacle
                    if math.hypot(x - cx, y - cy) <= radius:
                        blocked.append((row, col))
                        break
        return cls(resolution, resolution, blocked)

    def cell_center(self, cell: Cell, width: float, height: float) -> Tuple[float, float]:
        row, col = cell
        return (col + 0.5) / self.cols * width, (row + 0.5) / self.rows * height

    def locate(self, x: float, y: float, width: float, height: float) -> Cell:
        col = min(self.cols - 1, max(0, int(x / width * self.cols)))
        row = min(self.rows - 1, max(0, int(y / height * self.rows)))
        if (row, col) in self.blocked:
            # Step to the nearest free cell so a drone sitting on an obstacle
            # edge still has a start node.
            best = None
            best_dist = float("inf")
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in self.blocked:
                        continue
                    dist = abs(r - row) + abs(c - col)
                    if dist < best_dist:
                        best = (r, c)
                        best_dist = dist
            if best is None:
                raise RuntimeError("occupancy grid has no free cells")
            return best
        return row, col


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AntColonyPath:
    """Classic ant colony on a grid.

    Reference:
    Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization. MIT Press.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        start: Cell,
        goal: Cell,
        n_ants: int = 16,
        iterations: int = 20,
        alpha: float = 1.0,
        beta: float = 2.5,
        evaporation: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if not grid.contains(start) or not grid.contains(goal):
            raise ValueError("start and goal must be free cells")
        self.grid = grid
        self.start = start
        self.goal = goal
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.random = random.Random(seed)
        self.pheromone: Dict[Cell, float] = {}

    def _tau(self, cell: Cell) -> float:
        return self.pheromone.get(cell, 0.2)

    def _walk(self) -> Optional[List[Cell]]:
        cell = self.start
        path = [cell]
        visited = {cell}
        limit = self.grid.rows * self.grid.cols
        while cell != self.goal and len(path) <= limit:
            options = [nxt for nxt in self.grid.neighbors(cell) if nxt not in visited]
            if not options:
                return None
            weights = []
            for nxt in options:
                eta = 1.0 / (_manhattan(nxt, self.goal) + 0.1)
                weights.append((self._tau(nxt) ** self.alpha) * (eta ** self.beta))
            total = sum(weights)
            pick = self.random.random() * total
            acc = 0.0
            chosen = options[-1]
            for nxt, weight in zip(options, weights):
                acc += weight
                if acc >= pick:
                    chosen = nxt
                    break
            cell = chosen
            visited.add(cell)
            path.append(cell)
        if path[-1] != self.goal:
            return None
        return path

    def run(self) -> List[Cell]:
        best: Optional[List[Cell]] = None
        for _ in range(self.iterations):
            walks = []
            for _ant in range(self.n_ants):
                path = self._walk()
                if path is not None:
                    walks.append(path)
                    if best is None or len(path) < len(best):
                        best = path
            for cell in list(self.pheromone):
                self.pheromone[cell] *= 1.0 - self.evaporation
            for path in walks:
                deposit = 1.0 / len(path)
                for cell in path:
                    self.pheromone[cell] = self.pheromone.get(cell, 0.2) + deposit
        if best is not None:
            return best
        return self._greedy()

    def _greedy(self) -> List[Cell]:
        cell = self.start
        path = [cell]
        visited = {cell}
        while cell != self.goal and len(path) <= self.grid.rows * self.grid.cols:
            options = [nxt for nxt in self.grid.neighbors(cell) if nxt not in visited]
            if not options:
                break
            cell = min(options, key=lambda nxt: _manhattan(nxt, self.goal) - self._tau(nxt))
            visited.add(cell)
            path.append(cell)
        return path


class RiverFormationDynamics:
    """River formation dynamics on a grid altitude map.

    Reference:
    Rabanal, P., Rodríguez, I., & Rubio, F. (2007). Using river formation
    dynamics to design heuristic algorithms. Unconventional Computation.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        start: Cell,
        goal: Cell,
        n_drops: int = 24,
        iterations: int = 30,
        erosion: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if not grid.contains(start) or not grid.contains(goal):
            raise ValueError("start and goal must be free cells")
        self.grid = grid
        self.start = start
        self.goal = goal
        self.n_drops = n_drops
        self.iterations = iterations
        self.erosion = erosion
        self.random = random.Random(seed)
        self.altitude: Dict[Cell, float] = {}
        self.best: Optional[List[Cell]] = None
        plateau = float(grid.rows * grid.cols)
        for row in range(grid.rows):
            for col in range(grid.cols):
                cell = (row, col)
                self.altitude[cell] = 1e6 if cell in grid.blocked else plateau
        self.altitude[goal] = 0.0

    def _flow(self) -> None:
        """One drop walks downhill. A drop that reaches the goal carves a valley."""
        cell = self.start
        path = [cell]
        visited = {cell}
        limit = self.grid.rows * self.grid.cols
        while cell != self.goal and len(path) <= limit:
            options = [nxt for nxt in self.grid.neighbors(cell) if nxt not in visited]
            if not options:
                break
            cell = min(
                options,
                key=lambda nxt: self.altitude[nxt] + self.erosion * self.random.random(),
            )
            path.append(cell)
            visited.add(cell)
        if path[-1] != self.goal:
            return
        for index, step in enumerate(path):
            carved = float(len(path) - 1 - index)
            if carved < self.altitude[step]:
                self.altitude[step] = carved
        if self.best is None or len(path) < len(self.best):
            self.best = list(path)

    def run(self) -> List[Cell]:
        for _ in range(self.iterations):
            for _drop in range(self.n_drops):
                self._flow()
        descended = self._descend()
        if descended[-1] == self.goal:
            return descended
        if self.best is not None:
            return self.best
        return descended

    def _descend(self) -> List[Cell]:
        cell = self.start
        path = [cell]
        visited = {cell}
        while cell != self.goal and len(path) <= self.grid.rows * self.grid.cols:
            options = [nxt for nxt in self.grid.neighbors(cell) if nxt not in visited]
            if not options:
                break
            cell = min(options, key=lambda nxt: (self.altitude[nxt], _manhattan(nxt, self.goal)))
            visited.add(cell)
            path.append(cell)
        return path
