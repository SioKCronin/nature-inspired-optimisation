"""Marriage in Honey Bees.

Reference:
Abbass, H. A. (2001). MBO: marriage in honey bees optimization – a
haplometrosis polygynous swarming approach. Proceedings of the Congress on
Evolutionary Computation.
"""

from __future__ import annotations

from .base import PopulationOptimizer


class MarriageInHoneyBees(PopulationOptimizer):
    """A queen mates with drones along a flight of decaying speed."""

    def __init__(self, *args, flight_steps: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flight_steps = flight_steps

    def step(self) -> None:
        queen = min(self.population, key=lambda agent: agent.fitness)
        drones = [agent for agent in self.population if agent is not queen]
        speed = 1.0
        spermatheca = []
        for _ in range(self.flight_steps):
            if not drones:
                break
            drone = self.random.choice(drones)
            delta = abs(queen.fitness - drone.fitness) + 1e-9
            if self.rand() < 2.718 ** (-delta / speed):
                spermatheca.append(drone)
            speed *= 0.85
        children = [self.make_agent(queen.position[:], queen.fitness)]
        while len(children) < self.population_size:
            if spermatheca:
                drone = self.random.choice(spermatheca)
                child = self.mix(queen.position, drone.position)
            else:
                child = self.perturb(queen.position, 0.1)
            child = self.perturb(child, 0.02)
            position, fitness = self.place(child)
            children.append(self.make_agent(position, fitness))
        self.population = children
