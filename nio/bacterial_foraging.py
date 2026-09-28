"""Bacterial Foraging Optimization.

Reference:
Passino, K. M. (2002). Biomimicry of bacterial foraging for distributed
optimization and control. IEEE Control Systems Magazine, 22(3), 52–67.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BacterialForagingOptimization(PopulationOptimizer):
    """Chemotaxis with a swim, then reproduction of the healthiest half."""

    def __init__(self, *args, swim_length: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swim_length = swim_length

    def step(self) -> None:
        for agent in self.population:
            direction = [self.gauss() for _ in range(self.dimension)]
            norm = sum(v * v for v in direction) ** 0.5 + 1e-12
            step = 0.05 * (1.0 - 0.7 * self.progress())
            for _ in range(self.swim_length):
                trial = [
                    agent.position[i] + step * self.span(i) * direction[i] / norm
                    for i in range(self.dimension)
                ]
                position, fitness = self.place(trial)
                if fitness < agent.fitness:
                    self.adopt(agent, position, fitness)
                else:
                    break
        ranked = self.sorted_population()
        half = max(1, self.population_size // 2)
        parents = ranked[:half]
        children = []
        while len(children) < self.population_size:
            parent = parents[len(children) % half]
            if self.rand() < 0.1:
                position, fitness = self.place(self.random_position())
            else:
                position, fitness = self.place(self.perturb(parent.position, 0.02))
            children.append(self.make_agent(position, fitness))
        self.population = children
