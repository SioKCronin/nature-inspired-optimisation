"""Artificial Algae Algorithm.

Reference:
Uymaz, S. A., Tezel, G., & Yel, E. (2015). Artificial algae algorithm (AAA)
for numerical optimization. Applied Soft Computing, 31, 153–171.
"""

from __future__ import annotations

import math

from .base import PopulationOptimizer


class ArtificialAlgaeAlgorithm(PopulationOptimizer):
    """Colonies grow by helical motion, adapt, and replace starved cells."""

    def step(self) -> None:
        shear = 2.0 * math.pi * self.rand()
        for agent in self.population:
            friction = 0.3 * (1.0 - self.progress()) + 0.05
            trial = []
            for i in range(self.dimension):
                neighbor = self.population[self.random.randrange(self.population_size)].position[i]
                helical = friction * (neighbor - agent.position[i]) * math.cos(shear + i)
                trial.append(agent.position[i] + helical + 0.1 * (self.best_position[i] - agent.position[i]))
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
            elif self.rand() < 0.2:
                starved, starved_fitness = self.place(self.mix(agent.position, self.best_position, self.rand()))
                self.adopt(agent, starved, starved_fitness)
        worst = max(self.population, key=lambda item: item.fitness)
        if worst.fitness > self.best_fitness:
            position, fitness = self.place(self.perturb(self.best_position, 0.05))
            if fitness < worst.fitness:
                self.adopt(worst, position, fitness)
