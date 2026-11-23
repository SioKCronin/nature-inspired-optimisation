"""Altruistic Population Algorithm.

Reference:
Ouyang, H., Chen, J., Li, S., Xiang, J., & Zhan, Z.-H. (2023). Altruistic
population algorithm: a metaheuristic search algorithm for solving multimodal
multi-objective optimization problems. Mathematics and Computers in Simulation,
210, 296–319.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class AltruisticPopulationAlgorithm(PopulationOptimizer):
    """Egoists exploit the best point while altruists keep the search spread out."""

    def __init__(self, *args, altruist_rate: float = 0.3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.altruist_rate = altruist_rate

    def step(self) -> None:
        n_altruists = max(1, int(self.altruist_rate * self.population_size))
        ranked = self.sorted_population()
        altruists = {id(agent) for agent in ranked[-n_altruists:]}
        for agent in self.population:
            if id(agent) in altruists:
                farthest = max(self.population, key=lambda other: self.distance(agent.position, other.position))
                trial = self.mix(agent.position, farthest.position, 0.5)
                trial = self.perturb(trial, 0.08)
            else:
                guide = self.random.choice(ranked[: max(1, n_altruists)])
                trial = self.mix(agent.position, self.best_position, 0.4 + 0.4 * self.rand())
                if guide.fitness < agent.fitness:
                    trial = self.mix(trial, guide.position, 0.2)
            position, fitness = self.place(trial)
            if fitness <= agent.fitness or id(agent) in altruists:
                self.adopt(agent, position, fitness)
