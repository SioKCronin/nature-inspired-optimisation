"""Cuckoo Search.

Reference:
Yang, X.-S., & Deb, S. (2009). Cuckoo search via Lévy flights.
World Congress on Nature & Biologically Inspired Computing.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class CuckooSearch(PopulationOptimizer):
    """Lévy-flight nest replacement with a fraction of abandoned nests."""

    def __init__(self, *args, abandon: float = 0.25, step_scale: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.abandon = abandon
        self.step_scale = step_scale

    def step(self) -> None:
        for agent in list(self.population):
            trial = []
            for d in range(self.dimension):
                step = self.limited_levy(d, self.step_scale)
                bias = 0.2 * self.rand() * (self.best_position[d] - agent.position[d])
                trial.append(agent.position[d] + step + bias)
            position, fitness = self.place(trial)
            host = self.random.randrange(self.population_size)
            if fitness <= self.population[host].fitness:
                self.population[host] = self.make_agent(position, fitness)

        n_abandon = max(1, int(self.abandon * self.population_size))
        for agent in self.sorted_population()[-n_abandon:]:
            if self.rand() < 0.5:
                position, fitness = self.place(self.perturb(self.best_position, 0.08))
            else:
                position, fitness = self.place(self.random_position())
            self.adopt(agent, position, fitness)
