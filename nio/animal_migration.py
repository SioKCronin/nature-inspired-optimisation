"""Animal Migration Optimization.

Reference:
Li, X., Zhang, J., & Yin, M. (2014). Animal migration optimization: an
optimization algorithm inspired by animal migration behavior. Neural Computing
and Applications, 24, 1867–1877.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class AnimalMigrationOptimization(PopulationOptimizer):
    """Neighbors pull during migration; poorer animals are replaced."""

    def step(self) -> None:
        order = sorted(range(self.population_size), key=lambda i: self.population[i].fitness)
        for index, agent in enumerate(self.population):
            left = self.population[(index - 1) % self.population_size]
            right = self.population[(index + 1) % self.population_size]
            trial = []
            for d in range(self.dimension):
                neighbor = left if self.rand() < 0.5 else right
                trial.append(agent.position[d] + self.rand() * (neighbor.position[d] - agent.position[d]))
            position, fitness = self.place(trial)
            if fitness < agent.fitness:
                self.adopt(agent, position, fitness)
        n_replace = max(1, self.population_size // 3)
        probabilities = [1.0 - rank / self.population_size for rank in range(self.population_size)]
        for _ in range(n_replace):
            leaving = order[self.roulette([1.0 - p for p in probabilities])]
            donor = order[self.roulette(probabilities)]
            trial = self.mix(self.population[leaving].position, self.population[donor].position, self.rand())
            trial = self.perturb(trial, 0.05)
            position, fitness = self.place(trial)
            if fitness < self.population[leaving].fitness:
                self.adopt(self.population[leaving], position, fitness)
