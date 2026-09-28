"""Genetic Algorithm.

Reference:
Holland, J. H. (1975). Adaptation in Natural and Artificial Systems.
University of Michigan Press.
"""

from __future__ import annotations

from .base import Agent, PopulationOptimizer, Vector


class GeneticAlgorithm(PopulationOptimizer):
    """Real-coded genetic algorithm with tournament selection and elitism."""

    def __init__(
        self,
        *args,
        tournament_size: int = 3,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.1,
        elite_count: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_count = max(1, elite_count)

    def _tournament(self) -> Agent:
        size = min(self.tournament_size, self.population_size)
        indexes = self.distinct(size) if size < self.population_size else list(range(self.population_size))
        return min((self.population[i] for i in indexes), key=lambda agent: agent.fitness)

    def _mutate(self, position: Vector) -> Vector:
        child = list(position)
        for i in range(self.dimension):
            if self.rand() < self.mutation_rate:
                child[i] += self.gauss() * self.mutation_scale * self.span(i)
        return self.clamp(child)

    def step(self) -> None:
        ranked = self.sorted_population()
        children = [self.make_agent(agent.position[:], agent.fitness) for agent in ranked[: self.elite_count]]
        while len(children) < self.population_size:
            left = self._tournament()
            right = self._tournament()
            position = self._mutate(self.mix(left.position, right.position))
            placed, fitness = self.place(position)
            children.append(self.make_agent(placed, fitness))
        self.population = children
