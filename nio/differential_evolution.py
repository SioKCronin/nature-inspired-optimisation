"""Differential Evolution.

Reference:
Storn, R., & Price, K. (1997). Differential evolution – a simple and efficient
heuristic for global optimization over continuous spaces. Journal of Global
Optimization, 11, 341–359.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class DifferentialEvolution(PopulationOptimizer):
    """DE/rand/1/bin."""

    def __init__(self, *args, scale: float = 0.5, crossover: float = 0.9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.population_size < 4:
            raise ValueError("DifferentialEvolution requires population_size >= 4")
        self.scale = scale
        self.crossover = crossover

    def step(self) -> None:
        children = []
        for i, agent in enumerate(self.population):
            a, b, c = (self.population[j] for j in self.distinct(3, exclude=i))
            trial = []
            forced = self.random.randrange(self.dimension)
            for d in range(self.dimension):
                if d == forced or self.rand() < self.crossover:
                    value = a.position[d] + self.scale * (b.position[d] - c.position[d])
                    trial.append(self.clamp_scalar(value, d))
                else:
                    trial.append(agent.position[d])
            fitness = self.evaluate(trial)
            self.consider(trial, fitness)
            if fitness <= agent.fitness:
                children.append(self.make_agent(trial, fitness))
            else:
                children.append(agent)
        self.population = children
