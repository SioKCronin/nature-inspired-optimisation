"""Memetic Algorithm.

Reference:
Moscato, P. (1989). On evolution, search, optimization, genetic algorithms
and martial arts: Towards memetic algorithms. Caltech Concurrent Computation
Program, C3P Report 826.
"""

from __future__ import annotations

from .genetic import GeneticAlgorithm


class MemeticAlgorithm(GeneticAlgorithm):
    """Genetic search followed by a short local improvement of the elite."""

    def __init__(self, *args, local_steps: int = 4, local_scale: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.local_steps = local_steps
        self.local_scale = local_scale

    def step(self) -> None:
        super().step()
        elite = min(self.population, key=lambda agent: agent.fitness)
        scale = self.local_scale * (1.0 - 0.8 * self.progress())
        for _ in range(self.local_steps):
            candidate, fitness = self.place(self.perturb(elite.position, scale))
            if fitness < elite.fitness:
                self.adopt(elite, candidate, fitness)
