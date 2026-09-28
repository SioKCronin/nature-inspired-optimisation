"""Roach Infestation Optimization.

Reference:
Havens, T. C., Spain, C. J., Salmon, N. G., & Keller, J. M. (2008). Roach
infestation optimization. IEEE Swarm Intelligence Symposium.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class RoachInfestationOptimization(PopulationOptimizer):
    """Roaches mix personal memory, a social dark spot, and hungry search."""

    def __init__(self, *args, hunger_interval: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hunger_interval = hunger_interval

    def step(self) -> None:
        hungry = self.iteration % self.hunger_interval == 0
        for agent in self.population:
            personal = agent.best_position if agent.best_position is not None else agent.position
            velocity = []
            position = []
            for i in range(self.dimension):
                speed = (
                    0.7 * agent.velocity[i]
                    + 1.2 * self.rand() * (personal[i] - agent.position[i])
                    + 1.6 * self.rand() * (self.best_position[i] - agent.position[i])
                )
                if hungry:
                    speed += 0.3 * self.gauss() * self.span(i)
                limit = 0.2 * self.span(i)
                speed = max(-limit, min(limit, speed))
                velocity.append(speed)
                position.append(self.clamp_scalar(agent.position[i] + speed, i))
            agent.velocity = velocity
            fitness = self.evaluate(position)
            self.consider(position, fitness)
            self.adopt(agent, position, fitness)
            if hungry and self.rand() < 0.15:
                restarted, restarted_fitness = self.place(self.random_position())
                if restarted_fitness < agent.fitness:
                    self.adopt(agent, restarted, restarted_fitness)
