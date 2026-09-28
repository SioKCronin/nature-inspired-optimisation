"""Spiral Dynamic Algorithm.

Reference:
Tamura, K., & Yasuda, K. (2011). Spiral dynamics inspired optimization.
Journal of Advanced Computational Intelligence and Intelligent Informatics,
15(8), 1116–1122.
"""

from __future__ import annotations

import math

from .base import PopulationOptimizer


class SpiralDynamicAlgorithm(PopulationOptimizer):
    """Points spiral toward the current best on a shrinking radius and angle."""

    def step(self) -> None:
        radius = 0.95
        theta = math.pi / 4.0 * (1.0 - 0.5 * self.progress())
        centre = self.best_position
        for agent in self.population:
            trial = []
            for i in range(self.dimension):
                delta = agent.position[i] - centre[i]
                rotated = radius * (delta * math.cos(theta) - (0.0 if i + 1 == self.dimension else (agent.position[(i + 1) % self.dimension] - centre[(i + 1) % self.dimension]) * math.sin(theta)))
                trial.append(centre[i] + rotated)
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
