"""Biogeography-Based Optimization.

Reference:
Simon, D. (2008). Biogeography-based optimization. IEEE Transactions on
Evolutionary Computation, 12(6), 702–713.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BiogeographyBasedOptimization(PopulationOptimizer):
    """Habitat features migrate according to immigration and emigration rates."""

    def step(self) -> None:
        ranked = sorted(range(self.population_size), key=lambda i: self.population[i].fitness)
        rank_of = {index: order for order, index in enumerate(ranked)}
        n = self.population_size
        updated = [self.make_agent(agent.position[:], agent.fitness) for agent in self.population]
        for i, agent in enumerate(self.population):
            immigration = (rank_of[i] + 1) / n
            emigration_ranks = [n - rank_of[j] for j in range(n)]
            for d in range(self.dimension):
                if self.rand() < immigration:
                    donor = self.roulette(emigration_ranks)
                    updated[i].position[d] = self.population[donor].position[d]
                if self.rand() < 0.02:
                    updated[i].position[d] += self.gauss() * 0.05 * self.span(d)
            position, fitness = self.place(updated[i].position)
            updated[i] = self.make_agent(position, fitness)
        self.population = updated
