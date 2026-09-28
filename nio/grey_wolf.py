"""Grey Wolf Optimizer.

Reference:
Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
Advances in Engineering Software, 69, 46–61.
"""

from __future__ import annotations

from .base import PopulationOptimizer, Vector


class GreyWolfOptimizer(PopulationOptimizer):
    """Wolves encircle alpha, beta, and delta as the leadership hierarchy."""

    def _toward(self, leader: Vector, wolf: Vector, a: float) -> Vector:
        moved = []
        for i in range(self.dimension):
            r1 = self.rand()
            r2 = self.rand()
            capital_a = 2.0 * a * r1 - a
            capital_c = 2.0 * r2
            distance = abs(capital_c * leader[i] - wolf[i])
            moved.append(leader[i] - capital_a * distance)
        return moved

    def step(self) -> None:
        ranked = self.sorted_population()
        alpha = ranked[0].position
        beta = ranked[min(1, len(ranked) - 1)].position
        delta = ranked[min(2, len(ranked) - 1)].position
        a = 2.0 * (1.0 - self.progress())
        for agent in self.population:
            x1 = self._toward(alpha, agent.position, a)
            x2 = self._toward(beta, agent.position, a)
            x3 = self._toward(delta, agent.position, a)
            trial = [(x1[i] + x2[i] + x3[i]) / 3.0 for i in range(self.dimension)]
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
