"""Shuffled Frog Leaping Algorithm.

Reference:
Eusuff, M., Lansey, K., & Pasha, F. (2006). Shuffled frog-leaping algorithm:
a memetic meta-heuristic for discrete optimization. Engineering Optimization,
38(2), 129–154.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ShuffledFrogLeaping(PopulationOptimizer):
    """Frogs leap inside memeplexes, then the population is shuffled."""

    def __init__(self, *args, memeplexes: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.memeplexes = max(2, min(memeplexes, self.population_size))

    def _leap(self, worst_index: int, target_position) -> None:
        frog = self.population[worst_index]
        trial = [
            frog.position[i] + self.rand() * (target_position[i] - frog.position[i])
            for i in range(self.dimension)
        ]
        position, fitness = self.place(trial)
        if fitness < frog.fitness:
            self.adopt(frog, position, fitness)
            return
        position, fitness = self.place(self.perturb(self.best_position, 0.1))
        if fitness < frog.fitness:
            self.adopt(frog, position, fitness)
        else:
            position, fitness = self.place(self.random_position())
            self.adopt(frog, position, fitness)

    def step(self) -> None:
        ranked = sorted(range(self.population_size), key=lambda i: self.population[i].fitness)
        groups = [[] for _ in range(self.memeplexes)]
        for order, index in enumerate(ranked):
            groups[order % self.memeplexes].append(index)
        for group in groups:
            if len(group) < 2:
                continue
            worst = max(group, key=lambda i: self.population[i].fitness)
            best = min(group, key=lambda i: self.population[i].fitness)
            self._leap(worst, self.population[best].position)
