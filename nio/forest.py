"""Forest Optimization Algorithm.

Reference:
Ghaemi, M., & Feizi-Derakhshi, M.-R. (2014). Forest optimization algorithm.
Expert Systems with Applications, 41(15), 6676–6687.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ForestOptimizationAlgorithm(PopulationOptimizer):
    """Trees age, seed locally, and a few seeds are scattered globally."""

    def step(self) -> None:
        forest = []
        for agent in self.population:
            age = agent.velocity[0] if agent.velocity else 0.0
            if age > 6:
                position, fitness = self.place(self.random_position())
                sapling = self.make_agent(position, fitness)
                sapling.velocity[0] = 0.0
                forest.append(sapling)
                continue
            n_seeds = 2 if agent.fitness <= self.best_fitness * 1.5 else 1
            for _ in range(n_seeds):
                position, fitness = self.place(self.perturb(agent.position, 0.08 * (1.0 - 0.5 * self.progress())))
                seed = self.make_agent(position, fitness)
                seed.velocity[0] = 0.0
                forest.append(seed)
            agent.velocity[0] = age + 1.0
            forest.append(agent)
        for _ in range(max(1, self.population_size // 6)):
            position, fitness = self.place(self.random_position())
            seed = self.make_agent(position, fitness)
            seed.velocity[0] = 0.0
            forest.append(seed)
        self.population = sorted(forest, key=lambda tree: tree.fitness)[: self.population_size]
