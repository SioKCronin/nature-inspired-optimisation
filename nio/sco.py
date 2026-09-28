"""Social Cognitive Optimization.

Reference:
Xie, X.-F., Zhang, W.-J., & Yang, Z.-L. (2002). Social cognitive optimization
for nonlinear programming problems. Proceedings of the International Conference
on Machine Learning and Cybernetics.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class SocialCognitiveOptimization(PopulationOptimizer):
    """Agents learn from a shared library of good points and from neighbors."""

    def step(self) -> None:
        library = self.sorted_population()[: max(2, self.population_size // 4)]
        for agent in self.population:
            exemplar = self.random.choice(library)
            neighbor = self.population[self.random.randrange(self.population_size)]
            social = self.mix(exemplar.position, self.best_position, 0.5 + 0.5 * self.rand())
            trial = self.mix(agent.position, social, self.rand())
            if neighbor.fitness < agent.fitness:
                trial = self.mix(trial, neighbor.position, 0.3 * self.rand())
            if self.rand() < 0.15:
                trial = self.perturb(trial, 0.05)
            position, fitness = self.place(trial)
            if fitness <= agent.fitness:
                self.adopt(agent, position, fitness)
