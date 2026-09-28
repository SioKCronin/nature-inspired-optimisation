"""Honey-Bees Mating Optimization.

Reference:
Afshar, A., Bozorg Haddad, O., Marino, M. A., & Adams, B. J. (2007).
Honey-bee mating optimization (HBMO) algorithm for optimal reservoir operation.
Journal of the Franklin Institute, 344(5), 452–462.
"""

from __future__ import annotations

import math

from .base import PopulationOptimizer


class HoneyBeeMatingOptimization(PopulationOptimizer):
    """Queen mating flight, brood genotype, then worker local improvement."""

    def step(self) -> None:
        queen = min(self.population, key=lambda agent: agent.fitness)
        energy = 1.0
        speed = 1.0
        genome = []
        for drone in self.population:
            if drone is queen:
                continue
            probability = math.exp(-abs(queen.fitness - drone.fitness) / (speed + 1e-9))
            if self.rand() < probability * energy:
                genome.append(drone.position[:])
            energy *= 0.9
            speed *= 0.95
        brood = []
        while len(brood) < self.population_size - 1:
            if genome:
                donor = self.random.choice(genome)
                mask = self.rand()
                child = self.mix(queen.position, donor, mask)
            else:
                child = list(queen.position)
            child = self.perturb(child, 0.03 * (1.0 - 0.5 * self.progress()))
            position, fitness = self.place(child)
            brood.append(self.make_agent(position, fitness))
        workers = sorted(brood, key=lambda agent: agent.fitness)
        for agent in workers[: max(1, len(workers) // 3)]:
            improved, fitness = self.place(self.perturb(agent.position, 0.015))
            if fitness < agent.fitness:
                self.adopt(agent, improved, fitness)
        self.population = [self.make_agent(queen.position[:], queen.fitness)] + workers
        self.population = self.population[: self.population_size]
