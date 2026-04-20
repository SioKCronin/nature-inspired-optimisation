"""
Dragonfly Algorithm implementation with interception-aware prey pursuit.

Reference:
S. Mirjalili, "Dragonfly algorithm: a new meta-heuristic optimization technique
for solving single-objective, discrete, and multi-objective problems,"
Neural Computing and Applications, 2016.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Vector = List[float]
Objective = Callable[[Vector], float]


def rastrigin(position: Sequence[float]) -> float:
    """Default benchmark function (global minimum at 0)."""
    return 10 * len(position) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in position)


@dataclass
class Dragonfly:
    position: Vector
    step_vector: Vector
    fitness: float


class DragonflyAlgorithm:
    """Dragonfly optimizer with prey interception-aware target planning."""

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        inertia: float = 0.8,
        separation_weight: float = 0.1,
        alignment_weight: float = 0.1,
        cohesion_weight: float = 0.1,
        food_weight: float = 2.0,
        enemy_weight: float = 1.0,
        levy_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")

        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.inertia = inertia
        self.sep_w = separation_weight
        self.align_w = alignment_weight
        self.coh_w = cohesion_weight
        self.food_w = food_weight
        self.enemy_w = enemy_weight
        self.levy_scale = levy_scale
        self.random = random.Random(seed)

        self.population: List[Dragonfly] = []
        self.best: Dragonfly | None = None
        self.worst: Dragonfly | None = None
        self._previous_best_position: Vector | None = None

    def initialise(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            step_vector = [0.0] * self.dimension
            fitness = self.objective(position)
            self.population.append(Dragonfly(position=position, step_vector=step_vector, fitness=fitness))

        self.best = min(self.population, key=lambda d: d.fitness)
        self.worst = max(self.population, key=lambda d: d.fitness)
        self._previous_best_position = self.best.position[:] if self.best else None

    def _clip_position(self, position: Vector) -> Vector:
        clipped = position[:]
        for i, (lo, hi) in enumerate(self.bounds):
            if clipped[i] < lo:
                clipped[i] = lo
            elif clipped[i] > hi:
                clipped[i] = hi
        return clipped

    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _levy_flight(self) -> Vector:
        # Simple heavy-tailed jump generator for exploration.
        jump: Vector = []
        for _ in range(self.dimension):
            u = self.random.gauss(0.0, 1.0)
            v = self.random.gauss(0.0, 1.0)
            step = u / (abs(v) ** (1.0 / 1.5) + 1e-8)
            jump.append(self.levy_scale * step)
        return jump

    def _find_neighbors(self, index: int, radius: float) -> List[Dragonfly]:
        center = self.population[index]
        neighbors: List[Dragonfly] = []
        for j, candidate in enumerate(self.population):
            if j == index:
                continue
            if self._distance(center.position, candidate.position) <= radius:
                neighbors.append(candidate)
        return neighbors

    def _predict_prey_position(self, dragonfly: Dragonfly, prey_position: Vector, prey_velocity: Vector) -> Vector:
        # Estimate interception horizon as distance / current movement speed.
        speed = math.sqrt(sum(v * v for v in dragonfly.step_vector)) + 1e-8
        distance_to_prey = self._distance(dragonfly.position, prey_position)
        time_to_intercept = min(5.0, distance_to_prey / speed)
        predicted = [prey_position[i] + prey_velocity[i] * time_to_intercept for i in range(self.dimension)]
        return self._clip_position(predicted)

    def step(self, iteration: int, total_iterations: int) -> None:
        if self.best is None or self.worst is None:
            raise RuntimeError("Call initialise() before step().")

        # Expand neighborhood over time, shifting from local flocking to global consensus.
        span = [hi - lo for lo, hi in self.bounds]
        mean_span = sum(span) / max(len(span), 1)
        radius = mean_span * (0.1 + 0.8 * (iteration / max(total_iterations, 1)))

        prey_position = self.best.position[:]
        prev_best = self._previous_best_position if self._previous_best_position is not None else prey_position
        prey_velocity = [prey_position[i] - prev_best[i] for i in range(self.dimension)]
        enemy_position = self.worst.position[:]

        updated_positions: List[Vector] = []
        updated_steps: List[Vector] = []

        for i, dragonfly in enumerate(self.population):
            neighbors = self._find_neighbors(i, radius)

            if not neighbors:
                levy_jump = self._levy_flight()
                next_step = [self.inertia * dragonfly.step_vector[d] + levy_jump[d] for d in range(self.dimension)]
                next_position = [dragonfly.position[d] + next_step[d] for d in range(self.dimension)]
            else:
                n_count = float(len(neighbors))

                separation = [0.0] * self.dimension
                alignment = [0.0] * self.dimension
                cohesion_center = [0.0] * self.dimension
                for neighbor in neighbors:
                    for d in range(self.dimension):
                        separation[d] += dragonfly.position[d] - neighbor.position[d]
                        alignment[d] += neighbor.step_vector[d]
                        cohesion_center[d] += neighbor.position[d]

                alignment = [a / n_count for a in alignment]
                cohesion_center = [c / n_count for c in cohesion_center]
                cohesion = [cohesion_center[d] - dragonfly.position[d] for d in range(self.dimension)]

                predicted_prey = self._predict_prey_position(dragonfly, prey_position, prey_velocity)
                food_attraction = [predicted_prey[d] - dragonfly.position[d] for d in range(self.dimension)]
                enemy_distraction = [enemy_position[d] - dragonfly.position[d] for d in range(self.dimension)]

                next_step = []
                for d in range(self.dimension):
                    value = (
                        self.inertia * dragonfly.step_vector[d]
                        + self.sep_w * separation[d]
                        + self.align_w * alignment[d]
                        + self.coh_w * cohesion[d]
                        + self.food_w * food_attraction[d]
                        - self.enemy_w * enemy_distraction[d]
                    )
                    next_step.append(value)

                next_position = [dragonfly.position[d] + next_step[d] for d in range(self.dimension)]

            updated_steps.append(next_step)
            updated_positions.append(self._clip_position(next_position))

        for i, dragonfly in enumerate(self.population):
            dragonfly.step_vector = updated_steps[i]
            dragonfly.position = updated_positions[i]
            dragonfly.fitness = self.objective(dragonfly.position)

        self.best = min(self.population, key=lambda d: d.fitness)
        self.worst = max(self.population, key=lambda d: d.fitness)
        self._previous_best_position = prey_position

    def run(self, iterations: int = 100) -> Tuple[Vector, float]:
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        self.initialise()
        for iteration in range(iterations):
            if hasattr(self.objective, "update"):
                self.objective.update(iteration)
            self.step(iteration=iteration, total_iterations=iterations)

        if self.best is None:
            raise RuntimeError("Algorithm did not initialise best solution")
        return self.best.position[:], self.best.fitness


def main() -> None:
    optimizer = DragonflyAlgorithm(bounds=[(-5.12, 5.12)] * 5, population_size=40, seed=42)
    best_position, best_value = optimizer.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])


if __name__ == "__main__":
    main()
