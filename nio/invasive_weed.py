"""Invasive Weed Optimization.

Reference:
Mehrabian, A. R., & Lucas, C. (2006). A novel numerical optimization algorithm
inspired from weed colonization. Ecological Informatics, 1(4), 355–366.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class InvasiveWeedOptimization(PopulationOptimizer):
    """Fitter weeds shed more seeds, with a shrinking dispersal radius."""

    def __init__(self, *args, min_seeds: int = 0, max_seeds: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_seeds = min_seeds
        self.max_seeds = max_seeds

    def step(self) -> None:
        ranked = self.sorted_population()
        worst = ranked[-1].fitness
        best = ranked[0].fitness
        span = worst - best + 1e-12
        sigma = 0.3 * (1.0 - self.progress()) + 0.01
        colony = list(self.population)
        for agent in self.population:
            ratio = (worst - agent.fitness) / span
            n_seeds = int(round(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio))
            for _ in range(n_seeds):
                position, fitness = self.place(self.perturb(agent.position, sigma))
                colony.append(self.make_agent(position, fitness))
        self.population = sorted(colony, key=lambda agent: agent.fitness)[: self.population_size]
