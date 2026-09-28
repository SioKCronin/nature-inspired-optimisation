"""Social Spider Optimization.

Reference:
Cuevas, E., Cienfuegos, M., Zaldívar, D., & Pérez-Cisneros, M. (2013). A swarm
optimization algorithm inspired in the behavior of the social-spider.
Expert Systems with Applications, 40(16), 6374–6384.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class SocialSpiderOptimization(PopulationOptimizer):
    """Female attraction and male cooperative movement on a communal web."""

    def step(self) -> None:
        n_females = max(1, int(0.65 * self.population_size))
        females = self.population[:n_females]
        males = self.population[n_females:] or self.population[:1]
        median_male = sorted(males, key=lambda agent: agent.fitness)[len(males) // 2].fitness
        for agent in females:
            attract = [0.0] * self.dimension
            repel = [0.0] * self.dimension
            for other in self.population:
                if other is agent:
                    continue
                dist = self.distance(agent.position, other.position) + 1e-9
                vib = self.quality(other.fitness) / dist
                sign = 1.0 if other.fitness < agent.fitness else -1.0
                for d in range(self.dimension):
                    delta = vib * (other.position[d] - agent.position[d]) / dist
                    if sign > 0:
                        attract[d] += delta
                    else:
                        repel[d] += delta
            trial = [
                agent.position[d] + 0.3 * attract[d] - 0.1 * repel[d] + 0.05 * self.gauss() * self.span(d)
                for d in range(self.dimension)
            ]
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
        for agent in males:
            if agent.fitness <= median_male:
                trial = self.mix(agent.position, self.best_position, 0.4 * self.rand())
            else:
                centre = [
                    sum(male.position[d] for male in males) / len(males)
                    for d in range(self.dimension)
                ]
                trial = self.mix(agent.position, centre, 0.5)
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
        if females and males:
            female = min(females, key=lambda agent: agent.fitness)
            male = min(males, key=lambda agent: agent.fitness)
            child, fitness = self.place(self.mix(female.position, male.position))
            worst = max(self.population, key=lambda agent: agent.fitness)
            if fitness < worst.fitness:
                self.adopt(worst, child, fitness)
