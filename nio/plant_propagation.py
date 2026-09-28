"""Plant Propagation Algorithm.

Reference:
Salhi, A., & Fraga, E. S. (2011). Nature-inspired optimisation approaches and
the new plant propagation algorithm.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class PlantPropagationAlgorithm(PopulationOptimizer):
    """Normalized fitness sets both the number and the length of runners."""

    def step(self) -> None:
        ranked = self.sorted_population()
        worst = ranked[-1].fitness
        best = ranked[0].fitness
        span = worst - best + 1e-12
        nursery = [self.make_agent(ranked[0].position[:], ranked[0].fitness)]
        for agent in self.population:
            fitness_norm = (worst - agent.fitness) / span
            n_runners = max(1, int(1 + 3 * fitness_norm))
            for _ in range(n_runners):
                length = 0.5 * (1.0 - fitness_norm) * self.rand() + 0.01
                position, fitness = self.place(self.perturb(agent.position, length))
                nursery.append(self.make_agent(position, fitness))
        self.population = sorted(nursery, key=lambda agent: agent.fitness)[: self.population_size]
