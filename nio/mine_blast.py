"""Mine Blast Algorithm.

Reference:
Sadollah, A., Bahreininejad, A., Eskandar, H., & Hamdi, M. (2013). Mine blast
algorithm: a new population based algorithm for solving constrained engineering
optimization problems. Applied Soft Computing, 13(5), 2592–2612.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class MineBlastAlgorithm(PopulationOptimizer):
    """Shrapnel explodes around the current mine; the blast radius shrinks."""

    def step(self) -> None:
        explosion = 0.5 * (1.0 - self.progress()) + 0.02
        pieces = max(2, self.population_size // 2)
        for agent in self.population:
            best_piece = agent.position[:]
            best_fitness = agent.fitness
            direction = [self.gauss() for _ in range(self.dimension)]
            norm = sum(v * v for v in direction) ** 0.5 + 1e-12
            for piece in range(1, pieces + 1):
                trial = [
                    agent.position[d] + direction[d] / norm * explosion * self.span(d) * piece / pieces
                    for d in range(self.dimension)
                ]
                position, fitness = self.place(trial)
                if fitness < best_fitness:
                    best_piece = position
                    best_fitness = fitness
            self.adopt(agent, best_piece, best_fitness)
