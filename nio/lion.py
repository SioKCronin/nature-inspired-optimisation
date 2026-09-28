"""Lion Optimization Algorithm.

Reference:
Yazdani, M., & Jolai, F. (2016). Lion optimization algorithm (LOA): a
nature-inspired metaheuristic algorithm. Journal of Computational Design and
Engineering, 3(1), 24–36.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class LionOptimizationAlgorithm(PopulationOptimizer):
    """Pride hunters encircle prey while nomads search more widely."""

    def step(self) -> None:
        ranked = self.sorted_population()
        n_nomads = max(1, self.population_size // 5)
        pride = ranked[:-n_nomads] or ranked
        nomads = ranked[-n_nomads:]
        prey = self.best_position
        for lion in pride:
            trial = []
            for d in range(self.dimension):
                center = sum(member.position[d] for member in pride) / len(pride)
                trial.append(lion.position[d] + self.rand() * (prey[d] - center) * 0.5)
            position, fitness = self.place(trial)
            if fitness < lion.fitness:
                self.adopt(lion, position, fitness)
        for lion in nomads:
            if self.rand() < 0.5:
                trial = self.perturb(lion.position, 0.2 * (1.0 - 0.5 * self.progress()))
            else:
                trial = self.mix(lion.position, prey, self.rand())
            position, fitness = self.place(trial)
            if fitness < lion.fitness:
                self.adopt(lion, position, fitness)
        if pride and nomads:
            resident = self.random.choice(pride)
            nomad = min(nomads, key=lambda agent: agent.fitness)
            cub, fitness = self.place(self.mix(resident.position, nomad.position))
            weakest = max(self.population, key=lambda agent: agent.fitness)
            if fitness < weakest.fitness:
                self.adopt(weakest, cub, fitness)
