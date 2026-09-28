"""Glowworm Swarm Optimization.

Reference:
Krishnanand, K. N., & Ghose, D. (2009). Glowworm swarm optimization for
simultaneous capture of multiple local optima of multimodal functions.
Swarm Intelligence, 3, 87–124.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class GlowwormSwarmOptimization(PopulationOptimizer):
    """Luciferin-bearing agents move toward brighter neighbors in range."""

    def __init__(self, *args, sensor_range: float = 0.4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sensor_range = sensor_range
        self.luciferin = [1.0 for _ in range(self.population_size)]

    def initialise(self) -> None:
        super().initialise()
        self.luciferin = [self.quality(agent.fitness) for agent in self.population]

    def step(self) -> None:
        decay = 0.4
        gain = 0.6
        for i, agent in enumerate(self.population):
            self.luciferin[i] = (1.0 - decay) * self.luciferin[i] + gain * self.quality(agent.fitness)
        reach = self.sensor_range * (1.0 - 0.5 * self.progress()) * max(self.span(0), 1e-9)
        for i, agent in enumerate(self.population):
            neighbors = []
            weights = []
            for j, other in enumerate(self.population):
                if i == j:
                    continue
                if self.distance(agent.position, other.position) <= reach and self.luciferin[j] > self.luciferin[i]:
                    neighbors.append(other)
                    weights.append(self.luciferin[j] - self.luciferin[i])
            if neighbors:
                target = neighbors[self.roulette(weights)]
                step = 0.15
                trial = self.mix(agent.position, target.position, step)
            else:
                trial = self.perturb(agent.position, 0.05)
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
