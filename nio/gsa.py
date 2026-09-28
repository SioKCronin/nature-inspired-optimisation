"""Gravitational Search Algorithm.

Reference:
Rashedi, E., Nezamabadi-pour, H., & Saryazdi, S. (2009). GSA: a gravitational
search algorithm. Information Sciences, 179(13), 2232–2248.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class GravitationalSearchAlgorithm(PopulationOptimizer):
    """Masses attract each other; heavier masses pull harder as G decays."""

    def step(self) -> None:
        gravity = 10.0 * (1.0 - self.progress()) + 1e-6
        fitnesses = [agent.fitness for agent in self.population]
        best = min(fitnesses)
        worst = max(fitnesses)
        span = worst - best + 1e-12
        masses = [(worst - agent.fitness) / span for agent in self.population]
        total = sum(masses) + 1e-12
        masses = [mass / total for mass in masses]
        kbest = max(1, int(round(self.population_size * (1.0 - 0.8 * self.progress()))))
        heavy = sorted(range(self.population_size), key=lambda i: masses[i], reverse=True)[:kbest]
        for i, agent in enumerate(self.population):
            acceleration = [0.0] * self.dimension
            for j in heavy:
                if i == j:
                    continue
                dist = self.distance(agent.position, self.population[j].position) + 1e-9
                pull = gravity * masses[j] / dist
                for d in range(self.dimension):
                    acceleration[d] += self.rand() * pull * (
                        self.population[j].position[d] - agent.position[d]
                    )
            velocity = []
            position = []
            for d in range(self.dimension):
                speed = self.rand() * agent.velocity[d] + acceleration[d]
                limit = 0.2 * self.span(d)
                speed = max(-limit, min(limit, speed))
                velocity.append(speed)
                position.append(self.clamp_scalar(agent.position[d] + speed, d))
            agent.velocity = velocity
            fitness = self.evaluate(position)
            self.consider(position, fitness)
            self.adopt(agent, position, fitness)
