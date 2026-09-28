"""Spider Monkey Optimization.

Reference:
Bansal, J. C., Sharma, H., Jadon, S. S., & Clerc, M. (2014). Spider monkey
optimization algorithm for numerical optimization. Memetic Computing, 6, 31–47.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class SpiderMonkeyOptimization(PopulationOptimizer):
    """Local leaders pull small groups; a global leader perturbs the rest."""

    def step(self) -> None:
        groups = 2 if self.population_size < 12 else 3
        size = max(1, self.population_size // groups)
        for group in range(groups):
            members = self.population[group * size : (group + 1) * size] or self.population[-size:]
            local_leader = min(members, key=lambda agent: agent.fitness)
            for agent in members:
                if self.rand() < 0.6:
                    partner = self.random.choice(members)
                    dimension = self.random.randrange(self.dimension)
                    trial = list(agent.position)
                    trial[dimension] = (
                        agent.position[dimension]
                        + self.rand() * (local_leader.position[dimension] - agent.position[dimension])
                        + (self.rand() * 2 - 1) * (partner.position[dimension] - agent.position[dimension])
                    )
                    position, fitness = self.place(trial)
                    if fitness < agent.fitness:
                        self.adopt(agent, position, fitness)
        probabilities = [self.quality(agent.fitness) for agent in self.population]
        for index, agent in enumerate(self.population):
            if self.rand() < probabilities[index] / (max(probabilities) + 1e-12):
                dimension = self.random.randrange(self.dimension)
                trial = list(agent.position)
                trial[dimension] = agent.position[dimension] + self.rand() * (
                    self.best_position[dimension] - agent.position[dimension]
                )
                position, fitness = self.place(trial)
                if fitness < agent.fitness:
                    self.adopt(agent, position, fitness)
