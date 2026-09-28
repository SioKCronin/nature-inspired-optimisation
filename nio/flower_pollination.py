"""Flower Pollination Algorithm.

Reference:
Yang, X.-S. (2012). Flower pollination algorithm for global optimization.
Unconventional Computation and Natural Computation.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class FlowerPollinationAlgorithm(PopulationOptimizer):
    """Global Lévy pollination and local pollination between nearby flowers."""

    def __init__(self, *args, switch: float = 0.8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.switch = switch

    def step(self) -> None:
        for agent in self.population:
            if self.rand() < self.switch:
                trial = [
                    agent.position[d] + self.limited_levy(d) * (self.best_position[d] - agent.position[d])
                    for d in range(self.dimension)
                ]
            else:
                left, right = (self.population[i] for i in self.distinct(2))
                epsilon = self.rand()
                trial = [
                    agent.position[d] + epsilon * (left.position[d] - right.position[d])
                    for d in range(self.dimension)
                ]
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
