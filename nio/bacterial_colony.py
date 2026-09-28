"""Bacterial Colony Optimization.

Reference:
Niu, B., & Wang, H. (2012). Bacterial colony optimization. Discrete Dynamics
in Nature and Society.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BacterialColonyOptimization(PopulationOptimizer):
    """Chemotaxis, cell-to-cell communication, then a migration of weak cells."""

    def step(self) -> None:
        for agent in self.population:
            direction = [self.gauss() for _ in range(self.dimension)]
            norm = sum(v * v for v in direction) ** 0.5 + 1e-12
            step = 0.06 * (1.0 - 0.6 * self.progress())
            trial = [
                agent.position[i]
                + step * self.span(i) * direction[i] / norm
                + 0.2 * self.rand() * (self.best_position[i] - agent.position[i])
                for i in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
        for agent in self.population:
            partner = self.random.choice(self.population)
            if partner.fitness < agent.fitness:
                shared = self.mix(agent.position, partner.position, 0.3)
                position, fitness = self.place(shared)
                if fitness < agent.fitness:
                    self.adopt(agent, position, fitness)
        ranked = self.sorted_population()
        for agent in ranked[-(max(1, self.population_size // 5)) :]:
            if self.rand() < 0.5:
                position, fitness = self.place(self.random_position())
                self.adopt(agent, position, fitness)
