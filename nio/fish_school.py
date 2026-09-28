"""Artificial Fish School Algorithm.

Reference:
Li, X. L., Shao, Z. J., & Qian, J. X. (2002). An optimizing method based on
autonomous animats: fish-swarm algorithm. Systems Engineering — Theory &
Practice, 22(11), 32–38.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ArtificialFishSchool(PopulationOptimizer):
    """Prey, swarm, and follow behaviours inside a visual range."""

    def __init__(self, *args, visual: float = 0.3, crowd: float = 0.6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.visual = visual
        self.crowd = crowd

    def step(self) -> None:
        scale = self.visual * (1.0 - 0.5 * self.progress())
        for agent in self.population:
            neighbors = [
                other
                for other in self.population
                if other is not agent and self.distance(agent.position, other.position) < scale * max(self.span(0), 1.0)
            ]
            moved = False
            if neighbors:
                centre = [
                    sum(other.position[i] for other in neighbors) / len(neighbors)
                    for i in range(self.dimension)
                ]
                best_neighbor = min(neighbors, key=lambda other: other.fitness)
                crowded = len(neighbors) / self.population_size > self.crowd
                if not crowded and best_neighbor.fitness < agent.fitness:
                    trial = self.mix(agent.position, best_neighbor.position, self.rand() * 0.5)
                    moved = True
                elif not crowded:
                    trial = self.mix(agent.position, centre, self.rand() * 0.5)
                    moved = True
            if not moved:
                trial = self.perturb(agent.position, scale * 0.5)
            position, fitness = self.place(trial)
            if fitness <= agent.fitness:
                self.adopt(agent, position, fitness)
