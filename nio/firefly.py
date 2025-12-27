"""
Firefly Algorithm Implementation

Reference:
Xin-She Yang, "Firefly algorithms for multimodal optimization", 
in: Stochastic Algorithms: Foundations and Applications, SAGA 2009, 
Lecture Notes in Computer Sciences, Vol. 5792, pp. 169-178 (2009).
https://arxiv.org/pdf/1003.1466
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
class Firefly:
    """Represents a firefly in the Firefly Algorithm."""
    position: Vector
    intensity: float  # Objective function value (lower is better/brighter)


class FireflyAlgorithm:
    """
    Firefly Algorithm for multimodal optimization.
    
    The Firefly Algorithm is inspired by the flashing behavior of fireflies.
    Each firefly represents a solution, and brighter fireflies (better solutions)
    attract other fireflies based on light intensity (objective function value).
    """

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 40,
        alpha: float = 0.2,
        beta0: float = 1.0,
        gamma: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the Firefly Algorithm.
        
        Args:
            objective: Objective function to minimize
            bounds: Search space boundaries for each dimension
            population_size: Number of fireflies in the population
            alpha: Randomization parameter (controls exploration)
            beta0: Attractiveness at r=0 (maximum attractiveness)
            gamma: Light absorption coefficient (controls attractiveness decay)
            seed: Random seed for reproducibility
        """
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        
        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.random = random.Random(seed)
        
        self.population: List[Firefly] = []
        self.best: Firefly | None = None

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        """Initialize the firefly population."""
        self.population = []
        for _ in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            intensity = self.objective(position)
            firefly = Firefly(position, intensity)
            self.population.append(firefly)
        
        self.best = min(self.population, key=lambda f: f.intensity)

    # ------------------------------------------------------------------
    def _attractiveness(self, distance: float) -> float:
        """
        Calculate attractiveness based on distance.
        
        β(r) = β₀ * exp(-γ * r²)
        
        Args:
            distance: Euclidean distance between two fireflies
            
        Returns:
            Attractiveness value
        """
        return self.beta0 * math.exp(-self.gamma * distance * distance)
    
    def _distance(self, pos1: Vector, pos2: Vector) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(pos1, pos2)))

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Perform one iteration of the Firefly Algorithm."""
        if self.best is None:
            raise RuntimeError("Call initialise() before step().")
        
        new_positions = [firefly.position[:] for firefly in self.population]
        
        for i in range(self.population_size):
            firefly_i = self.population[i]
            total_movement = [0.0] * self.dimension
            moved = False
            
            # Move firefly i toward all brighter fireflies
            for j in range(self.population_size):
                firefly_j = self.population[j]
                
                # If firefly j is brighter (lower intensity = better) than firefly i
                if firefly_j.intensity < firefly_i.intensity:
                    # Calculate distance between fireflies i and j
                    distance = self._distance(firefly_i.position, firefly_j.position)
                    
                    # Calculate attractiveness
                    beta = self._attractiveness(distance)
                    
                    # Accumulate movement toward firefly j
                    # x_i = x_i + β * exp(-γ * r²) * (x_j - x_i)
                    for dim in range(self.dimension):
                        total_movement[dim] += beta * (firefly_j.position[dim] - firefly_i.position[dim])
                    moved = True
            
            # Update position with movement and randomization
            if moved:
                # Add randomization: α * ε
                for dim in range(self.dimension):
                    randomization = self.alpha * (self.random.random() - 0.5) * 2  # Uniform in [-alpha, alpha]
                    new_positions[i][dim] += total_movement[dim] + randomization
            else:
                # If no brighter firefly, add randomization to prevent stagnation
                for dim in range(self.dimension):
                    randomization = self.alpha * (self.random.random() - 0.5) * 2
                    new_positions[i][dim] += randomization
            
            # Apply bounds constraints
            for dim, (lo, hi) in enumerate(self.bounds):
                if new_positions[i][dim] < lo:
                    new_positions[i][dim] = lo
                elif new_positions[i][dim] > hi:
                    new_positions[i][dim] = hi
        
        # Evaluate new positions and update population
        for i, new_pos in enumerate(new_positions):
            new_intensity = self.objective(new_pos)
            self.population[i].position = new_pos
            self.population[i].intensity = new_intensity
            
            # Update best solution
            if self.best is None or new_intensity < self.best.intensity:
                self.best = Firefly(new_pos[:], new_intensity)

    # ------------------------------------------------------------------
    def run(self, iterations: int = 100) -> Tuple[Vector, float]:
        """Run the Firefly Algorithm optimization.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            Tuple of (best_position, best_intensity)
        """
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        
        self.initialise()
        for iteration in range(iterations):
            # Update stateful objectives (e.g., contracting optimum)
            if hasattr(self.objective, 'update'):
                self.objective.update(iteration)
            self.step()
        
        if self.best is None:
            raise RuntimeError("Algorithm did not initialise best solution")
        return self.best.position[:], self.best.intensity


def main() -> None:
    """Example usage of the Firefly Algorithm."""
    fa = FireflyAlgorithm(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0,
        seed=42,
    )
    best_position, best_value = fa.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])


if __name__ == "__main__":
    main()

