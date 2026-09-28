"""Gases Brownian Motion Optimization.

Reference:
Abdechiri, M., Meybodi, M. R., & Bahrami, H. (2013). Gases Brownian motion
optimization: an algorithm for optimization (GBMO). Applied Soft Computing,
13(5), 2932–2946.
"""

from __future__ import annotations

import math

from .base import PopulationOptimizer


class GasesBrownianMotionOptimization(PopulationOptimizer):
    """Molecules mix turbulent motion toward heavier gases with Brownian jitter."""

    def step(self) -> None:
        temperature = 1.0 - self.progress()
        for agent in self.population:
            trial = []
            for d in range(self.dimension):
                turbulent = self.rand() * (self.best_position[d] - agent.position[d])
                brownian = math.sqrt(temperature + 1e-9) * self.gauss() * 0.05 * self.span(d)
                trial.append(agent.position[d] + turbulent + brownian)
            position, fitness = self.place(trial)
            if fitness < agent.fitness or self.rand() < temperature * 0.1:
                self.adopt(agent, position, fitness)
