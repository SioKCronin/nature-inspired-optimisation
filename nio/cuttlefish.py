"""Cuttlefish Algorithm.

Reference:
Eesa, A. S., Brifcani, A. M. A., & Orman, Z. (2013). Cuttlefish algorithm – a
novel bio-inspired optimization algorithm. International Journal of Scientific
& Engineering Research, 4(9).
"""

from __future__ import annotations

from .base import PopulationOptimizer


class CuttlefishAlgorithm(PopulationOptimizer):
    """Reflection and visibility operators split the population into four cases."""

    def step(self) -> None:
        ranked = self.sorted_population()
        quarter = max(1, self.population_size // 4)
        groups = [
            ranked[:quarter],
            ranked[quarter : 2 * quarter],
            ranked[2 * quarter : 3 * quarter],
            ranked[3 * quarter :],
        ]
        best = self.best_position
        for index, group in enumerate(groups):
            for agent in group:
                trial = []
                for d in range(self.dimension):
                    reflection = self.rand() * best[d]
                    visibility = self.rand() * (best[d] - agent.position[d])
                    if index == 0:
                        trial.append(reflection + visibility)
                    elif index == 1:
                        trial.append(reflection - visibility)
                    elif index == 2:
                        trial.append(self.rand() * agent.position[d] + visibility)
                    else:
                        trial.append(self.random.uniform(*self.bounds[d]))
                position, fitness = self.place(trial)
                if fitness < agent.fitness:
                    self.adopt(agent, position, fitness)
