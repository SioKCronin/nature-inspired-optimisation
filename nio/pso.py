"""Particle Swarm Optimization.

Reference:
Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
Proceedings of ICNN'95.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ParticleSwarmOptimization(PopulationOptimizer):
    """Canonical PSO with inertia weight, minimizing."""

    def __init__(self, *args, inertia: float = 0.72, cognitive: float = 1.49, social: float = 1.49, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def step(self) -> None:
        for agent in self.population:
            velocity = []
            position = []
            for i in range(self.dimension):
                personal = agent.best_position[i] if agent.best_position is not None else agent.position[i]
                speed = (
                    self.inertia * agent.velocity[i]
                    + self.cognitive * self.rand() * (personal - agent.position[i])
                    + self.social * self.rand() * (self.best_position[i] - agent.position[i])
                )
                limit = 0.2 * self.span(i)
                speed = max(-limit, min(limit, speed))
                velocity.append(speed)
                position.append(self.clamp_scalar(agent.position[i] + speed, i))
            agent.velocity = velocity
            fitness = self.evaluate(position)
            self.consider(position, fitness)
            self.adopt(agent, position, fitness)
