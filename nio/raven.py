"""The Raven Roosting Optimisation Algorithm.

Reference:
Brabazon, A., Cui, W., & O'Neill, M. (2016). The raven roosting optimisation
algorithm. Soft Computing, 20, 525–545.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class RavenRoostingOptimization(PopulationOptimizer):
    """Ravens leave a roost to forage, then a fraction follow the best finder."""

    def __init__(self, *args, follow_rate: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.follow_rate = follow_rate

    def step(self) -> None:
        roost = self.best_position
        for agent in self.population:
            if self.rand() < self.follow_rate:
                leader = min(self.population, key=lambda item: item.fitness)
                trial = self.mix(agent.position, leader.position, self.rand())
            else:
                trial = self.perturb(roost, 0.15 * (1.0 - 0.7 * self.progress()) + 0.02)
                if self.rand() < 0.3:
                    trial = self.mix(trial, agent.position, 0.5)
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
            else:
                returned, returned_fitness = self.place(self.perturb(roost, 0.04))
                if returned_fitness < agent.fitness:
                    self.adopt(agent, returned, returned_fitness)
