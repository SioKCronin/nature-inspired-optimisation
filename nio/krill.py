"""Krill Herd.

Reference:
Gandomi, A. H., & Alavi, A. H. (2012). Krill herd: a new bio-inspired
optimization algorithm. Communications in Nonlinear Science and Numerical
Simulation, 17(12), 4831–4845.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class KrillHerd(PopulationOptimizer):
    """Motion induced by neighbors, foraging toward food, and physical diffusion."""

    def step(self) -> None:
        sensing = 0.35 * (1.0 - 0.5 * self.progress()) * max(self.span(0), 1e-9)
        food = self.best_position
        for agent in self.population:
            induced = [0.0] * self.dimension
            count = 0
            for other in self.population:
                if other is agent:
                    continue
                dist = self.distance(agent.position, other.position)
                if dist < sensing:
                    weight = self.quality(other.fitness) / (dist + 1e-9)
                    for d in range(self.dimension):
                        induced[d] += weight * (other.position[d] - agent.position[d])
                    count += 1
            if count:
                induced = [value / count for value in induced]
            trial = []
            for d in range(self.dimension):
                forage = 0.4 * (food[d] - agent.position[d])
                diffusion = 0.02 * (1.0 - self.progress()) * self.gauss() * self.span(d)
                trial.append(agent.position[d] + 0.15 * induced[d] + forage + diffusion)
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
