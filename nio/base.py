"""Shared population-optimizer contract for the nio corpus.

New algorithms subclass :class:`PopulationOptimizer` and implement :meth:`step`.
The public surface matches the rest of the library: construct with an objective,
bounds, population size, and seed; call ``run(iterations)`` to receive
``(best_position, best_value)``. Every optimizer minimizes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

Vector = List[float]
Objective = Callable[[Sequence[float]], float]


def sphere(position: Sequence[float]) -> float:
    """Sphere function. Global minimum is 0 at the origin."""
    return sum(x * x for x in position)


@dataclass
class Agent:
    """One member of a population search."""

    position: Vector
    fitness: float
    velocity: Vector = field(default_factory=list)
    best_position: Optional[Vector] = None
    best_fitness: float = float("inf")


class PopulationOptimizer:
    """Base class for continuous population metaheuristics."""

    def __init__(
        self,
        objective: Objective = sphere,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        seed: int | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if len(bounds) == 0:
            raise ValueError("bounds must be non-empty")
        self.objective = objective
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.dimension = len(self.bounds)
        self.population_size = population_size
        self.random = random.Random(seed)
        self.population: List[Agent] = []
        self.best_position: Vector = []
        self.best_fitness: float = float("inf")
        self.iteration = 0
        self.iterations = 1

    def random_position(self) -> Vector:
        return [self.random.uniform(lo, hi) for lo, hi in self.bounds]

    def random_velocity(self) -> Vector:
        return [
            self.random.uniform(-0.1 * (hi - lo), 0.1 * (hi - lo))
            for lo, hi in self.bounds
        ]

    def span(self, index: int) -> float:
        lo, hi = self.bounds[index]
        return hi - lo

    def clamp_scalar(self, value: float, index: int) -> float:
        lo, hi = self.bounds[index]
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    def clamp(self, position: Sequence[float]) -> Vector:
        return [self.clamp_scalar(float(position[i]), i) for i in range(self.dimension)]

    def evaluate(self, position: Sequence[float]) -> float:
        value = float(self.objective(list(position)))
        if math.isnan(value) or math.isinf(value):
            return 1e12
        return value

    def consider(self, position: Sequence[float], fitness: float) -> None:
        if fitness < self.best_fitness:
            self.best_position = list(position)
            self.best_fitness = fitness

    def place(self, position: Sequence[float]) -> Tuple[Vector, float]:
        """Clamp, evaluate, and record a candidate."""
        clamped = self.clamp(position)
        fitness = self.evaluate(clamped)
        self.consider(clamped, fitness)
        return clamped, fitness

    def make_agent(self, position: Vector, fitness: float) -> Agent:
        return Agent(
            position=list(position),
            fitness=fitness,
            velocity=self.random_velocity(),
            best_position=list(position),
            best_fitness=fitness,
        )

    def initialise(self) -> None:
        self.population = []
        self.best_fitness = float("inf")
        self.best_position = []
        self.iteration = 0
        for _ in range(self.population_size):
            position, fitness = self.place(self.random_position())
            self.population.append(self.make_agent(position, fitness))

    def step(self) -> None:
        raise NotImplementedError

    def run(self, iterations: int = 100) -> Tuple[Vector, float]:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.initialise()
        for iteration in range(iterations):
            self.iteration = iteration
            if hasattr(self.objective, "update"):
                self.objective.update(iteration)
            self.step()
        if not self.best_position:
            raise RuntimeError("Algorithm did not initialise best solution")
        return self.best_position[:], self.best_fitness

    def progress(self) -> float:
        """Fraction of the run completed, in ``[0, 1]``."""
        if self.iterations <= 1:
            return 1.0
        return self.iteration / float(self.iterations - 1)

    def rand(self) -> float:
        return self.random.random()

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self.random.gauss(mu, sigma)

    def levy(self, beta: float = 1.5) -> float:
        """One draw from a Lévy distribution via Mantegna's algorithm."""
        numerator = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        denominator = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
        sigma = (numerator / denominator) ** (1.0 / beta)
        u = self.gauss(0.0, sigma)
        v = self.gauss(0.0, 1.0)
        return u / ((abs(v) ** (1.0 / beta)) + 1e-12)

    def limited_levy(self, index: int, scale: float = 0.05) -> float:
        step = self.levy() * scale * self.span(index)
        limit = 0.25 * self.span(index)
        if step > limit:
            return limit
        if step < -limit:
            return -limit
        return step

    def distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(self.dimension)))

    def perturb(self, position: Sequence[float], scale: float) -> Vector:
        return self.clamp(
            [
                position[i] + self.gauss(0.0, scale) * self.span(i)
                for i in range(self.dimension)
            ]
        )

    def mix(self, left: Sequence[float], right: Sequence[float], rate: float | None = None) -> Vector:
        """Arithmetic crossover. ``rate`` is the weight on ``right``."""
        if rate is None:
            rate = self.rand()
        return self.clamp(
            [
                (1.0 - rate) * left[i] + rate * right[i]
                for i in range(self.dimension)
            ]
        )

    def sorted_population(self) -> List[Agent]:
        return sorted(self.population, key=lambda agent: agent.fitness)

    def distinct(self, k: int, exclude: int | None = None) -> List[int]:
        pool = [i for i in range(self.population_size) if i != exclude]
        if k > len(pool):
            raise ValueError("population is too small for this operator")
        return self.random.sample(pool, k)

    def roulette(self, weights: Sequence[float]) -> int:
        total = sum(max(0.0, w) for w in weights)
        if total <= 0.0:
            return self.random.randrange(len(weights))
        pick = self.rand() * total
        acc = 0.0
        for i, weight in enumerate(weights):
            acc += max(0.0, weight)
            if acc >= pick:
                return i
        return len(weights) - 1

    def quality(self, fitness: float) -> float:
        """Positive selection weight. Smaller fitness is more attractive."""
        if fitness >= 0.0:
            return 1.0 / (1.0 + fitness)
        return 1.0 + abs(fitness)

    def adopt(self, agent: Agent, position: Sequence[float], fitness: float) -> None:
        agent.position = list(position)
        agent.fitness = fitness
        if fitness < agent.best_fitness:
            agent.best_position = list(position)
            agent.best_fitness = fitness
