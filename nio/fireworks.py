"""Fireworks Algorithm.

Reference:
Tan, Y., & Zhu, Y. (2010). Fireworks algorithm for optimization.
International Conference in Swarm Intelligence.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class FireworksAlgorithm(PopulationOptimizer):
    """Explosions shed sparks; better fireworks explode more tightly."""

    def step(self) -> None:
        ranked = self.sorted_population()
        best = ranked[0].fitness
        worst = ranked[-1].fitness
        sparks = list(self.population)
        amplitude_scale = 0.4 * (1.0 - 0.85 * self.progress()) + 0.01
        for agent in self.population:
            quality = (worst - agent.fitness) / (worst - best + 1e-12)
            n_sparks = 2 + int(6 * quality)
            amplitude = amplitude_scale * (agent.fitness - best + 1e-6) / (worst - best + 1e-12)
            amplitude = min(0.5, max(0.01, amplitude))
            for _ in range(n_sparks):
                trial = list(agent.position)
                for d in range(self.dimension):
                    if self.rand() < 0.5:
                        trial[d] += (self.rand() * 2.0 - 1.0) * amplitude * self.span(d)
                position, fitness = self.place(trial)
                sparks.append(self.make_agent(position, fitness))
        for _ in range(max(1, self.population_size // 5)):
            position, fitness = self.place(self.perturb(self.best_position, 0.05))
            sparks.append(self.make_agent(position, fitness))
        self.population = sorted(sparks, key=lambda agent: agent.fitness)[: self.population_size]
