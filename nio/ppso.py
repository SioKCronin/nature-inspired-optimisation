"""
Proactive Particle Swarm Optimization (PPSO) Implementation

Reference:
Cheng, R., & Jin, Y. (2015). A social learning particle swarm optimization algorithm 
for scalable optimization. Information Sciences, 291, 43-60.

The algorithm uses proactive particles that use knowledge gain metrics to explore
regions with low sample density, combined with traditional reactive particles.
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
class Particle:
    """Represents a particle in PPSO."""
    position: Vector
    velocity: Vector
    best_position: Vector
    best_fitness: float
    fitness: float
    is_proactive: bool = False
    knowledge_gain: float = 0.0


class PPSO:
    """Proactive Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        proactive_ratio: float = 0.25,
        c1: float = 2.0,
        c2: float = 2.0,
        w: float = 0.9,
        knowledge_method: str = "gaussian",
        kernel_width: float = 1.0,
        exploration_weight: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the PPSO algorithm.

        Args:
            objective: Objective function to minimize
            bounds: Search space boundaries for each dimension
            population_size: Total number of particles
            proactive_ratio: Ratio of proactive particles (0.2-0.3 recommended)
            c1: Cognitive coefficient
            c2: Social coefficient
            w: Inertia weight
            knowledge_method: Method for calculating knowledge gain ("gaussian", "inverse_distance")
            kernel_width: Width parameter for knowledge gain calculation
            exploration_weight: Weight for exploration component
            seed: Random seed for reproducibility
        """
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0.0 <= proactive_ratio <= 1.0:
            raise ValueError("proactive_ratio must be between 0 and 1")
        
        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.proactive_ratio = proactive_ratio
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.knowledge_method = knowledge_method
        self.kernel_width = kernel_width
        self.exploration_weight = exploration_weight
        self.random = random.Random(seed)

        # Calculate number of proactive particles
        self.n_proactive = int(population_size * proactive_ratio)
        self.n_reactive = population_size - self.n_proactive

        # Sample history for knowledge gain calculation
        self.sample_history: List[Tuple[Vector, float]] = []

        self.population: List[Particle] = []
        self.best: Particle | None = None

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        """Initialize the population of particles."""
        self.population = []
        self.sample_history = []

        for i in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            velocity = [
                self.random.uniform(-abs(hi - lo), abs(hi - lo))
                for lo, hi in self.bounds
            ]
            fitness = self.objective(position)
            is_proactive = i < self.n_proactive

            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position[:],
                best_fitness=fitness,
                fitness=fitness,
                is_proactive=is_proactive,
            )
            self.population.append(particle)
            self.sample_history.append((position[:], fitness))

        self.best = min(self.population, key=lambda p: p.best_fitness)

    # ------------------------------------------------------------------
    def _calculate_knowledge_gain(self, position: Vector) -> float:
        """Calculate knowledge gain at a given position.
        
        Higher knowledge gain indicates unexplored regions with potential
        for discovering better solutions.
        """
        if len(self.sample_history) < 2:
            return 1.0  # Maximum gain for unexplored regions

        if self.knowledge_method == "gaussian":
            return self._gaussian_knowledge_gain(position)
        elif self.knowledge_method == "inverse_distance":
            return self._inverse_distance_knowledge_gain(position)
        else:
            raise ValueError(f"Unknown knowledge method: {self.knowledge_method}")

    def _gaussian_knowledge_gain(self, position: Vector) -> float:
        """Gaussian Process-inspired knowledge gain."""
        # Calculate minimum distance to all previous samples
        min_distance = float('inf')
        for sample_pos, _ in self.sample_history:
            distance = math.sqrt(
                sum((position[i] - sample_pos[i]) ** 2 for i in range(self.dimension))
            )
            min_distance = min(min_distance, distance)

        # Gaussian kernel: higher gain for larger distances
        knowledge_gain = math.exp(-(min_distance ** 2) / (2 * self.kernel_width ** 2))
        
        # Normalize by sample density
        sample_density = len(self.sample_history) / (4 * math.pi * self.kernel_width ** 2)
        knowledge_gain *= (1.0 / (1.0 + sample_density))
        
        return knowledge_gain

    def _inverse_distance_knowledge_gain(self, position: Vector) -> float:
        """Inverse distance-based knowledge gain."""
        min_distance = float('inf')
        for sample_pos, _ in self.sample_history:
            distance = math.sqrt(
                sum((position[i] - sample_pos[i]) ** 2 for i in range(self.dimension))
            )
            min_distance = min(min_distance, distance)

        if min_distance < 1e-10:
            return 0.0

        return 1.0 / (1.0 + min_distance)

    def _update_proactive_particle(self, particle: Particle, iteration: int, max_iterations: int) -> None:
        """Update a proactive particle using knowledge gain."""
        # Calculate knowledge gain at current position
        particle.knowledge_gain = self._calculate_knowledge_gain(particle.position)

        # Adaptive exploration weight (decreases over time)
        progress = iteration / max_iterations
        adaptive_exploration = self.exploration_weight * (1.0 - progress)
        adaptive_exploration += particle.knowledge_gain * 0.5  # Bonus for high knowledge gain

        # Sample exploration directions and choose best
        n_directions = 5
        best_direction = None
        best_kg = -1.0

        for _ in range(n_directions):
            # Generate random direction
            direction = [self.random.gauss(0, 1) for _ in range(self.dimension)]
            magnitude = math.sqrt(sum(d * d for d in direction))
            if magnitude > 0:
                direction = [d / magnitude for d in direction]

            # Test position in this direction
            step_size = 0.1
            test_position = [
                particle.position[i] + step_size * direction[i]
                for i in range(self.dimension)
            ]
            # Apply bounds
            for i, (lo, hi) in enumerate(self.bounds):
                test_position[i] = max(lo, min(hi, test_position[i]))

            kg = self._calculate_knowledge_gain(test_position)
            if kg > best_kg:
                best_kg = kg
                best_direction = direction

        # Update velocity with exploration component
        if best_direction and self.best:
            cognitive = [
                self.c1 * self.random.random() * (particle.best_position[i] - particle.position[i])
                for i in range(self.dimension)
            ]
            social = [
                self.c2 * self.random.random() * (self.best.position[i] - particle.position[i])
                for i in range(self.dimension)
            ]
            exploration = [
                adaptive_exploration * best_direction[i] * particle.knowledge_gain
                for i in range(self.dimension)
            ]

            for i in range(self.dimension):
                particle.velocity[i] = (
                    self.w * particle.velocity[i] +
                    cognitive[i] +
                    social[i] +
                    exploration[i]
                )
        else:
            # Fallback to standard PSO update
            self._update_reactive_particle(particle)

    def _update_reactive_particle(self, particle: Particle) -> None:
        """Update a reactive particle using standard PSO."""
        if not self.best:
            return

        for i in range(self.dimension):
            cognitive = self.c1 * self.random.random() * (particle.best_position[i] - particle.position[i])
            social = self.c2 * self.random.random() * (self.best.position[i] - particle.position[i])
            particle.velocity[i] = self.w * particle.velocity[i] + cognitive + social

    def _move_particle(self, particle: Particle) -> None:
        """Move particle and apply bounds."""
        for i in range(self.dimension):
            particle.position[i] += particle.velocity[i]
            lo, hi = self.bounds[i]
            if particle.position[i] < lo:
                particle.position[i] = lo
                particle.velocity[i] *= -0.5  # Bounce back
            elif particle.position[i] > hi:
                particle.position[i] = hi
                particle.velocity[i] *= -0.5  # Bounce back

    # ------------------------------------------------------------------
    def step(self, iteration: int = 0, max_iterations: int = 100) -> None:
        """Perform one optimization step.
        
        Args:
            iteration: Current iteration number (for adaptive exploration)
            max_iterations: Maximum iterations (for adaptive exploration)
        """
        if self.best is None:
            raise RuntimeError("Call initialise() before step().")

        for particle in self.population:
            # Update particle based on type
            if particle.is_proactive:
                # Proactive particles use knowledge gain with adaptive exploration
                self._update_proactive_particle(particle, iteration, max_iterations)
            else:
                # Reactive particles use standard PSO
                self._update_reactive_particle(particle)

            # Move particle
            self._move_particle(particle)

            # Evaluate fitness
            particle.fitness = self.objective(particle.position)

            # Update personal best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position[:]

            # Update global best
            if particle.best_fitness < self.best.best_fitness:
                self.best = Particle(
                    position=particle.position[:],
                    velocity=particle.velocity[:],
                    best_position=particle.best_position[:],
                    best_fitness=particle.best_fitness,
                    fitness=particle.fitness,
                    is_proactive=particle.is_proactive,
                    knowledge_gain=particle.knowledge_gain,
                )

            # Add to sample history (for knowledge gain calculation)
            self.sample_history.append((particle.position[:], particle.fitness))

    # ------------------------------------------------------------------
    def run(self, iterations: int = 100) -> Tuple[Vector, float]:
        """Run the optimization algorithm for specified iterations."""
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        self.initialise()
        for iteration in range(iterations):
            # Update stateful objectives (e.g., contracting optimum)
            if hasattr(self.objective, 'update'):
                self.objective.update(iteration)
            
            self.step(iteration, iterations)
        
        if self.best is None:
            raise RuntimeError("Algorithm did not initialise best solution")
        return self.best.best_position[:], self.best.best_fitness


def main() -> None:
    """Example usage of the PPSO algorithm."""
    optimizer = PPSO(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        proactive_ratio=0.25,
        seed=42
    )
    best_position, best_value = optimizer.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])
    print(f"Best found by: {'Proactive' if optimizer.best.is_proactive else 'Reactive'} particle")


if __name__ == "__main__":
    main()

