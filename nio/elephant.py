"""Elephant Herding Optimization.

Reference:
Wang, G.-G., Deb, S., & Coelho, L. d. S. (2015). Elephant herding optimization.
International Symposium on Computational Intelligence and Informatics.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ElephantHerdingOptimization(PopulationOptimizer):
    """Clan members move toward their matriarch; the worst elephant separates."""

    def __init__(self, *args, n_clans: int = 3, alpha: float = 0.5, beta: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_clans = max(1, min(n_clans, self.population_size))
        self.alpha = alpha
        self.beta = beta

    def step(self) -> None:
        indexes = list(range(self.population_size))
        self.random.shuffle(indexes)
        clans = [indexes[i :: self.n_clans] for i in range(self.n_clans)]
        for clan in clans:
            if not clan:
                continue
            matriarch = min(clan, key=lambda i: self.population[i].fitness)
            centre = [
                sum(self.population[i].position[d] for i in clan) / len(clan)
                for d in range(self.dimension)
            ]
            for i in clan:
                elephant = self.population[i]
                if i == matriarch:
                    trial = [
                        self.beta * centre[d] + (1.0 - self.beta) * elephant.position[d]
                        for d in range(self.dimension)
                    ]
                else:
                    trial = [
                        elephant.position[d]
                        + self.alpha * (self.population[matriarch].position[d] - elephant.position[d]) * self.rand()
                        for d in range(self.dimension)
                    ]
                position, fitness = self.place(trial)
                self.adopt(elephant, position, fitness)
            worst = max(clan, key=lambda i: self.population[i].fitness)
            position, fitness = self.place(
                [
                    self.bounds[d][0]
                    + (self.bounds[d][1] - self.bounds[d][0]) * self.rand()
                    for d in range(self.dimension)
                ]
            )
            if fitness < self.population[worst].fitness or self.rand() < 0.5:
                self.adopt(self.population[worst], position, fitness)
