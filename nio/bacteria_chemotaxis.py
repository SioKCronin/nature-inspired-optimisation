"""Bacterial Chemotaxis.

Reference:
Muller, S. D., Marchetto, J., Airaghi, S., & Koumoutsakos, P. (2002).
Optimization based on bacterial chemotaxis. IEEE Transactions on Evolutionary
Computation, 6(1), 16–29.
"""

from __future__ import annotations

import math

from .base import PopulationOptimizer


class BacterialChemotaxis(PopulationOptimizer):
    """Tumble toward an estimated gradient, then take a chemotactic step."""

    def step(self) -> None:
        step = 0.08 * (1.0 - 0.75 * self.progress())
        for agent in self.population:
            gradient = []
            for i in range(self.dimension):
                delta = 0.02 * self.span(i)
                plus = list(agent.position)
                minus = list(agent.position)
                plus[i] = self.clamp_scalar(plus[i] + delta, i)
                minus[i] = self.clamp_scalar(minus[i] - delta, i)
                gradient.append((self.evaluate(plus) - self.evaluate(minus)) / (2.0 * delta))
            norm = math.sqrt(sum(g * g for g in gradient)) + 1e-12
            trial = [
                agent.position[i] - step * self.span(i) * gradient[i] / norm + self.gauss(0.0, 0.01) * self.span(i)
                for i in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            if fitness <= agent.fitness or self.rand() < 0.1:
                self.adopt(agent, position, fitness)
