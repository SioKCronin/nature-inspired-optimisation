"""Black Hole Algorithm.

Reference:
Hatamlou, A. (2013). Black hole: a new heuristic optimization approach for
data clustering. Information Sciences, 222, 175–184.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BlackHoleAlgorithm(PopulationOptimizer):
    """Stars move toward the black hole and are swallowed inside its horizon."""

    def step(self) -> None:
        black_hole = min(self.population, key=lambda agent: agent.fitness)
        total = sum(abs(agent.fitness) for agent in self.population) + 1e-12
        horizon = abs(black_hole.fitness) / total
        reach = horizon * max(self.span(i) for i in range(self.dimension))
        for agent in self.population:
            if agent is black_hole:
                continue
            trial = [
                agent.position[i] + self.rand() * (black_hole.position[i] - agent.position[i])
                for i in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
            if self.distance(agent.position, black_hole.position) < reach:
                restarted, restarted_fitness = self.place(self.random_position())
                self.adopt(agent, restarted, restarted_fitness)
