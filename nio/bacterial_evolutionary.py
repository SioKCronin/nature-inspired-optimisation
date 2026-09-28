"""Bacterial Evolutionary Algorithm.

Reference:
Nawa, N. E., & Furuhashi, T. (1999). Fuzzy system parameters discovery by
bacterial evolutionary algorithm. IEEE Transactions on Fuzzy Systems, 7(5),
608–616.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BacterialEvolutionaryAlgorithm(PopulationOptimizer):
    """Bacterial mutation of clones, then gene transfer from better cells."""

    def step(self) -> None:
        mutated = []
        for agent in self.population:
            champion = self.make_agent(agent.position[:], agent.fitness)
            for d in range(self.dimension):
                for _ in range(3):
                    trial = list(champion.position)
                    trial[d] += self.gauss() * 0.1 * self.span(d) * (1.0 - 0.5 * self.progress())
                    position, fitness = self.place(trial)
                    if fitness < champion.fitness:
                        champion = self.make_agent(position, fitness)
            mutated.append(champion)
        self.population = mutated
        ranked = sorted(range(self.population_size), key=lambda i: self.population[i].fitness)
        transfers = max(1, self.population_size // 4)
        for _ in range(transfers):
            donor = ranked[self.random.randrange(max(1, len(ranked) // 2))]
            receiver = ranked[-(self.random.randrange(max(1, len(ranked) // 2)) + 1)]
            dimension = self.random.randrange(self.dimension)
            trial = list(self.population[receiver].position)
            trial[dimension] = self.population[donor].position[dimension]
            position, fitness = self.place(trial)
            if fitness < self.population[receiver].fitness:
                self.adopt(self.population[receiver], position, fitness)
