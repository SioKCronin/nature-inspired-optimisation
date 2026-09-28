"""Water Wave Optimization.

Reference:
Zheng, Y.-J. (2015). Water wave optimization: a new nature-inspired
metaheuristic. Computers & Operations Research, 55, 1–11.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class WaterWaveOptimization(PopulationOptimizer):
    """Propagation across a wavelength, refraction when a wave stalls, then breaking."""

    def step(self) -> None:
        ranked = self.sorted_population()
        worst = ranked[-1].fitness
        best = ranked[0].fitness
        for agent in self.population:
            wavelength = (agent.fitness - best + 1e-9) / (worst - best + 1e-9)
            wavelength = min(1.0, max(1e-3, wavelength))
            trial = [
                agent.position[d] + self.rand() * (-1.0 if self.rand() < 0.5 else 1.0) * wavelength * self.span(d)
                for d in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
            else:
                refracted = self.mix(agent.position, self.best_position, 0.5)
                position, fitness = self.place(self.perturb(refracted, 0.02))
                if fitness < agent.fitness:
                    self.adopt(agent, position, fitness)
        breaker = min(self.population, key=lambda agent: agent.fitness)
        for _ in range(max(1, self.dimension)):
            position, fitness = self.place(self.perturb(breaker.position, 0.01))
            if fitness < breaker.fitness:
                self.adopt(breaker, position, fitness)
