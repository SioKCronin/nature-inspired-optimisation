"""Artificial immune system via clonal selection.

Reference:
De Castro, L. N., & Von Zuben, F. J. (2002). Learning and optimization using
the clonal selection principle. IEEE Transactions on Evolutionary Computation,
6(3), 239–251.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ClonalSelection(PopulationOptimizer):
    """CLONALG: clone the best antibodies and hypermutate them."""

    def __init__(self, *args, clone_factor: int = 4, replace_rate: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clone_factor = clone_factor
        self.replace_rate = replace_rate

    def step(self) -> None:
        ranked = self.sorted_population()
        n_keep = max(1, self.population_size // 2)
        clones = []
        for rank, agent in enumerate(ranked[:n_keep]):
            n_clones = max(1, self.clone_factor - rank)
            scale = 0.15 * (rank + 1) / n_keep
            for _ in range(n_clones):
                position, fitness = self.place(self.perturb(agent.position, scale))
                clones.append(self.make_agent(position, fitness))
        n_random = max(1, int(self.replace_rate * self.population_size))
        for _ in range(n_random):
            position, fitness = self.place(self.random_position())
            clones.append(self.make_agent(position, fitness))
        pool = ranked + clones
        self.population = sorted(pool, key=lambda agent: agent.fitness)[: self.population_size]
