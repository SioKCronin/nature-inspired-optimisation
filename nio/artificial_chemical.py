"""Artificial Chemical Reaction Optimization.

Reference:
Alatas, B. (2011). ACROA: artificial chemical reaction optimization algorithm
for global optimization. Expert Systems with Applications, 38(10), 13170–13180.

Listed in the README as the artificial chemical process algorithm.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class ArtificialChemicalReactionOptimization(PopulationOptimizer):
    """Decomposition, synthesis, and collisions on a population of reactants."""

    def step(self) -> None:
        reactants = list(self.population)
        for agent in list(self.population):
            if self.rand() < 0.25:
                left, fitness_left = self.place(self.perturb(agent.position, 0.1))
                right, fitness_right = self.place(self.perturb(agent.position, 0.1))
                reactants.append(self.make_agent(left, fitness_left))
                reactants.append(self.make_agent(right, fitness_right))
            else:
                partner = self.random.choice(self.population)
                product, fitness = self.place(self.mix(agent.position, partner.position))
                reactants.append(self.make_agent(product, fitness))
        for agent in self.population:
            bounced = [
                self.bounds[d][0] + self.bounds[d][1] - agent.position[d]
                if self.rand() < 0.1
                else agent.position[d]
                for d in range(self.dimension)
            ]
            position, fitness = self.place(bounced)
            reactants.append(self.make_agent(position, fitness))
        self.population = sorted(reactants, key=lambda agent: agent.fitness)[: self.population_size]
