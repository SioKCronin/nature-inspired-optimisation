"""Strawberry Algorithm.

Reference:
Salhi, A., & Fraga, E. S. (2011). Nature-inspired optimisation approaches and
the new plant propagation algorithm. Proceedings of the International
Conference on Numerical Analysis and Applied Mathematics.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class StrawberryAlgorithm(PopulationOptimizer):
    """Good plants send short runners; poor plants send long ones."""

    def step(self) -> None:
        ranked = self.sorted_population()
        worst = ranked[-1].fitness
        best = ranked[0].fitness
        span = worst - best + 1e-12
        colony = []
        for agent in self.population:
            quality = (worst - agent.fitness) / span
            n_runners = 1 + int(3 * quality)
            length = 0.05 + 0.4 * (1.0 - quality)
            for _ in range(n_runners):
                position, fitness = self.place(self.perturb(agent.position, length))
                colony.append(self.make_agent(position, fitness))
            colony.append(self.make_agent(agent.position[:], agent.fitness))
        self.population = sorted(colony, key=lambda item: item.fitness)[: self.population_size]
