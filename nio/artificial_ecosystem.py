"""Artificial Ecosystem Algorithm.

Reference:
Zhao, W., Wang, L., & Zhang, Z. (2020). Artificial ecosystem-based optimization:
a novel nature-inspired meta-heuristic algorithm. Neural Computing and
Applications, 32, 9383–9425.

The production, consumption, and decomposition operators follow that paper's
roles even though the README lists the method as an artificial ecosystem algorithm.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ArtificialEcosystemAlgorithm(PopulationOptimizer):
    """Producers seek the best, consumers eat neighbors, decomposers recycle."""

    def step(self) -> None:
        producer = min(self.population, key=lambda agent: agent.fitness)
        if self.rand() < 0.5:
            trial = [
                (1.0 - self.rand()) * producer.position[d] + self.rand() * self.bounds[d][1]
                for d in range(self.dimension)
            ]
        else:
            trial = [
                (1.0 - self.rand()) * producer.position[d] + self.rand() * self.bounds[d][0]
                for d in range(self.dimension)
            ]
        position, fitness = self.place(trial)
        if fitness < producer.fitness:
            self.adopt(producer, position, fitness)
        ranked = self.sorted_population()
        for index, agent in enumerate(ranked[1:], start=1):
            prey = ranked[self.random.randrange(index)]
            consumption = self.rand()
            trial = [
                agent.position[d] + consumption * (prey.position[d] - agent.position[d])
                for d in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
        weight = 3.0
        decomposer = self.best_position
        for agent in self.population:
            trial = []
            for d in range(self.dimension):
                d1 = self.rand() * (decomposer[d] - agent.position[d])
                d2 = self.rand() * (agent.position[d] - decomposer[d])
                trial.append(agent.position[d] + weight * (d1 + d2) * 0.1)
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
