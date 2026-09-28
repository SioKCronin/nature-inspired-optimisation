"""Bull Optimization Algorithm.

Reference:
The bull optimization algorithm is a population search in which bulls charge
the current best location and a fraction of the herd is scattered to keep
the search from collapsing. The README lists it alongside the other
herd metaheuristics in this corpus.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BullOptimizationAlgorithm(PopulationOptimizer):
    """Bulls charge the lead animal; restless bulls are sent back to pasture."""

    def step(self) -> None:
        for agent in self.population:
            charge = 0.3 + 0.7 * self.rand()
            trial = [
                agent.position[d] + charge * (self.best_position[d] - agent.position[d])
                + 0.05 * (1.0 - self.progress()) * self.gauss() * self.span(d)
                for d in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
        restless = self.sorted_population()[-max(1, self.population_size // 4) :]
        for agent in restless:
            if self.rand() < 0.4:
                position, fitness = self.place(self.random_position())
                self.adopt(agent, position, fitness)
