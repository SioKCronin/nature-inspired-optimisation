"""Artificial Bee Colony.

Reference:
Karaboga, D. (2005). An idea based on honey bee swarm for numerical
optimization. Technical Report TR06, Erciyes University.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ArtificialBeeColony(PopulationOptimizer):
    """Employed bees, onlookers, and scouts sharing a food-source memory."""

    def __init__(self, *args, limit: int = 12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.trials = [0 for _ in range(self.population_size)]

    def initialise(self) -> None:
        super().initialise()
        self.trials = [0 for _ in range(self.population_size)]

    def _neighbor(self, index: int) -> None:
        agent = self.population[index]
        other = self.population[self.distinct(1, exclude=index)[0]]
        dimension = self.random.randrange(self.dimension)
        trial = list(agent.position)
        phi = self.rand() * 2.0 - 1.0
        trial[dimension] = agent.position[dimension] + phi * (agent.position[dimension] - other.position[dimension])
        position, fitness = self.place(trial)
        if fitness < agent.fitness:
            self.adopt(agent, position, fitness)
            self.trials[index] = 0
        else:
            self.trials[index] += 1

    def step(self) -> None:
        for index in range(self.population_size):
            self._neighbor(index)
        weights = [self.quality(agent.fitness) for agent in self.population]
        for _ in range(self.population_size):
            self._neighbor(self.roulette(weights))
        for index, agent in enumerate(self.population):
            if self.trials[index] >= self.limit:
                position, fitness = self.place(self.random_position())
                self.adopt(agent, position, fitness)
                self.trials[index] = 0
