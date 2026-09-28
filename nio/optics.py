"""Optics Inspired Optimization.

Reference:
Kashan, A. H. (2015). A new metaheuristic for optimization: optics inspired
optimization (OIO). Computers & Operations Research, 55, 99–125.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class OpticsInspiredOptimization(PopulationOptimizer):
    """Artificial images form by reflecting a ray through a spherical mirror."""

    def step(self) -> None:
        for agent in self.population:
            mirror = self.population[self.random.randrange(self.population_size)]
            if mirror.fitness > agent.fitness:
                mirror = min(self.population, key=lambda item: item.fitness)
            trial = []
            for d in range(self.dimension):
                object_distance = abs(agent.position[d] - mirror.position[d]) + 1e-9
                radius = 0.2 * self.span(d) * (1.0 - 0.5 * self.progress()) + 1e-6
                focal = radius / 2.0
                image = 1.0 / (1.0 / focal - 1.0 / object_distance + 1e-9)
                magnification = image / object_distance
                trial.append(mirror.position[d] - magnification * (agent.position[d] - mirror.position[d]))
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
