"""Bird Mating Optimizer.

Reference:
Askarzadeh, A. (2014). Bird mating optimizer: an optimization algorithm inspired
by bird mating strategies. Communications in Nonlinear Science and Numerical
Simulation, 19(4), 1213–1228.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class BirdMatingOptimizer(PopulationOptimizer):
    """Monogamy, polygyny, and promiscuity produce the next brood."""

    def step(self) -> None:
        ranked = self.sorted_population()
        n = self.population_size
        brood = []
        for index, bird in enumerate(ranked):
            if index < n // 2:
                mate = ranked[min(n - 1, index + 1 + self.random.randrange(max(1, n // 4)))]
                child = self.mix(bird.position, mate.position, 0.3 + 0.4 * self.rand())
            elif index < 3 * n // 4:
                mates = ranked[: max(2, n // 5)]
                child = list(bird.position)
                for mate in mates[:3]:
                    child = self.mix(child, mate.position, self.rand() / 3.0)
            else:
                child = self.mix(bird.position, self.best_position, self.rand())
                child = self.perturb(child, 0.08)
            position, fitness = self.place(child)
            brood.append(self.make_agent(position, fitness))
        self.population = brood
