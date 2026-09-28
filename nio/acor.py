"""Ant Colony Optimization for continuous domains.

Reference:
Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
European Journal of Operational Research, 185(3), 1155–1173.

Discrete path-finding ants live in :mod:`nio.pathfinding`.
"""

from __future__ import annotations

import math

from .base import PopulationOptimizer


class AntColonyOptimization(PopulationOptimizer):
    """ACOR: sample new ants from a ranked Gaussian kernel over the archive."""

    def __init__(self, *args, intensification: float = 0.5, deviation: float = 0.85, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.intensification = intensification
        self.deviation = deviation

    def step(self) -> None:
        ranked = self.sorted_population()
        weights = []
        size = len(ranked)
        q = self.intensification
        for rank in range(size):
            weights.append(
                (1.0 / (q * size * math.sqrt(2.0 * math.pi)))
                * math.exp(-(rank ** 2) / (2.0 * q * q * size * size))
            )
        ants = []
        for _ in range(self.population_size):
            chosen = ranked[self.roulette(weights)]
            trial = []
            for d in range(self.dimension):
                sigma = 0.0
                for other in ranked:
                    sigma += abs(other.position[d] - chosen.position[d])
                sigma = self.deviation * sigma / (size - 1 if size > 1 else 1)
                trial.append(self.gauss(chosen.position[d], sigma + 1e-9))
            position, fitness = self.place(trial)
            ants.append(self.make_agent(position, fitness))
        archive = ranked + ants
        self.population = sorted(archive, key=lambda agent: agent.fitness)[: self.population_size]
