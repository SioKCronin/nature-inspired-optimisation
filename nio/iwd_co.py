"""
Intelligent Water Drops - Continuous Optimization (IWD-CO) Implementation

Reference:
H. Shah-Hosseini, "The intelligent water drops algorithm: a nature-inspired swarm-based 
optimization algorithm", International Journal of Bio-Inspired Computation, 2009.
Adapted for continuous optimization problems.

The algorithm simulates water drops flowing through a landscape, where:
- Water drops move toward areas with less soil (better solutions)
- Velocity increases in low-soil areas
- Soil is picked up from visited positions and deposited at worse positions
- Movement is guided by the best solution and local soil conditions

Enhanced with multiple liquid and soil types, each with distinct properties affecting
behavior (viscosity, carrying capacity, resistance, etc.).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Sequence, Tuple

Vector = List[float]
Objective = Callable[[Vector], float]


def rastrigin(position: Sequence[float]) -> float:
    """Default benchmark function (global minimum at 0)."""
    return 10 * len(position) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in position)


class LiquidType(Enum):
    """Types of liquids with different physical properties."""
    H2O = {
        "viscosity": 1.0,      # Baseline viscosity
        "density": 1.0,        # Baseline density
        "carrying_capacity": 1.0,  # Soil carrying capacity
        "velocity_multiplier": 1.0,  # Base velocity
    }
    OIL = {
        "viscosity": 0.5,      # Lower viscosity (flows faster)
        "density": 0.8,        # Lower density
        "carrying_capacity": 0.6,  # Carries less soil
        "velocity_multiplier": 1.3,  # Faster movement
    }
    ALCOHOL = {
        "viscosity": 0.3,      # Very low viscosity
        "density": 0.8,        # Lower density
        "carrying_capacity": 0.4,  # Low carrying capacity
        "velocity_multiplier": 1.5,  # Very fast movement
    }
    GLYCEROL = {
        "viscosity": 2.0,      # High viscosity (flows slower)
        "density": 1.3,        # Higher density
        "carrying_capacity": 1.5,  # Carries more soil
        "velocity_multiplier": 0.7,  # Slower movement
    }
    MERCURY = {
        "viscosity": 0.15,     # Very low viscosity
        "density": 13.5,       # Very high density
        "carrying_capacity": 0.3,  # Low carrying capacity
        "velocity_multiplier": 1.8,  # Very fast, but heavy
    }
    HONEY = {
        "viscosity": 5.0,      # Very high viscosity
        "density": 1.4,        # Higher density
        "carrying_capacity": 2.0,  # High carrying capacity
        "velocity_multiplier": 0.4,  # Very slow movement
    }

    def get_property(self, prop: str) -> float:
        """Get a property value for this liquid type."""
        return self.value[prop]


class SoilType(Enum):
    """Types of soil with different properties affecting interaction with liquids."""
    SAND = {
        "resistance": 0.5,         # Low resistance (easy to move through)
        "pickup_difficulty": 0.3,  # Easy to pick up
        "deposition_rate": 0.8,    # Deposits easily
        "erosion_rate": 1.2,       # Erodes easily
    }
    CLAY = {
        "resistance": 2.0,         # High resistance (hard to move through)
        "pickup_difficulty": 1.5,  # Hard to pick up
        "deposition_rate": 1.5,    # Deposits slowly
        "erosion_rate": 0.5,       # Erodes slowly
    }
    SILT = {
        "resistance": 0.8,         # Moderate resistance
        "pickup_difficulty": 0.6,  # Moderate pickup difficulty
        "deposition_rate": 1.0,     # Standard deposition
        "erosion_rate": 1.0,       # Standard erosion
    }
    GRAVEL = {
        "resistance": 1.5,         # High resistance
        "pickup_difficulty": 2.0,  # Very hard to pick up
        "deposition_rate": 0.5,    # Deposits quickly
        "erosion_rate": 0.3,       # Very slow erosion
    }
    LOAM = {
        "resistance": 0.6,         # Low resistance
        "pickup_difficulty": 0.4,  # Easy to pick up
        "deposition_rate": 1.2,    # Deposits well
        "erosion_rate": 1.1,       # Moderate erosion
    }
    PEAT = {
        "resistance": 0.4,         # Very low resistance
        "pickup_difficulty": 0.2,  # Very easy to pick up
        "deposition_rate": 1.5,    # Deposits very easily
        "erosion_rate": 1.5,       # Erodes very easily
    }

    def get_property(self, prop: str) -> float:
        """Get a property value for this soil type."""
        return self.value[prop]


@dataclass
class WaterDrop:
    """Represents a water drop in the IWD-CO algorithm."""
    position: Vector
    velocity: Vector
    soil: Dict[SoilType, float] = field(default_factory=dict)  # Soil amounts by type
    liquid_type: LiquidType = LiquidType.H2O
    fitness: float = 0.0


class IWDCO:
    """Intelligent Water Drops algorithm for Continuous Optimization.
    
    Enhanced with multiple liquid and soil types, each affecting behavior differently.
    """

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        initial_velocity: float = 1.0,
        initial_soil: float = 0.0,
        velocity_update_parameter: float = 0.01,
        soil_update_parameter: float = 0.01,
        alpha: float = 1.0,  # Soil update coefficient
        beta: float = 1.0,   # Velocity update coefficient
        liquid_types: Sequence[LiquidType] | None = None,  # Available liquid types
        soil_types: Sequence[SoilType] | None = None,  # Available soil types
        liquid_distribution: str = "uniform",  # "uniform" or "random"
        seed: int | None = None,
    ) -> None:
        """Initialize the IWD-CO algorithm.

        Args:
            objective: Objective function to minimize
            bounds: Search space boundaries for each dimension
            population_size: Number of water drops
            initial_velocity: Initial velocity for water drops
            initial_soil: Initial soil amount for water drops
            velocity_update_parameter: Parameter controlling velocity updates
            soil_update_parameter: Parameter controlling soil updates
            alpha: Coefficient for soil update
            beta: Coefficient for velocity update
            liquid_types: Available liquid types (default: all)
            soil_types: Available soil types (default: all)
            liquid_distribution: How to assign liquid types ("uniform" or "random")
            seed: Random seed for reproducibility
        """
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.initial_velocity = initial_velocity
        self.initial_soil = initial_soil
        self.velocity_update_parameter = velocity_update_parameter
        self.soil_update_parameter = soil_update_parameter
        self.alpha = alpha
        self.beta = beta
        self.random = random.Random(seed)

        # Material type configuration
        self.liquid_types = liquid_types if liquid_types else list(LiquidType)
        self.soil_types = soil_types if soil_types else list(SoilType)
        self.liquid_distribution = liquid_distribution

        self.population: List[WaterDrop] = []
        self.best: WaterDrop | None = None
        # Soil landscape: track soil by type (simplified as global soil per type)
        self.global_soil: Dict[SoilType, float] = {soil_type: 0.0 for soil_type in self.soil_types}

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        """Initialize the population of water drops."""
        self.population = []
        self.global_soil = {soil_type: 0.0 for soil_type in self.soil_types}

        for i in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            # Initialize velocity as a small random vector
            velocity = [
                self.random.uniform(-self.initial_velocity, self.initial_velocity)
                for _ in range(self.dimension)
            ]
            
            # Assign liquid type
            if self.liquid_distribution == "uniform":
                liquid_type = self.liquid_types[i % len(self.liquid_types)]
            else:  # random
                liquid_type = self.random.choice(self.liquid_types)
            
            # Initialize soil by type
            soil = {soil_type: self.initial_soil for soil_type in self.soil_types}
            
            fitness = self.objective(position)
            drop = WaterDrop(position, velocity, soil, liquid_type, fitness)
            self.population.append(drop)

        self.best = min(self.population, key=lambda d: d.fitness)

    # ------------------------------------------------------------------
    def _update_velocity(self, drop: WaterDrop, target_position: Sequence[float]) -> None:
        """Update velocity based on soil content, target position, and material properties.
        
        Velocity increases when moving toward areas with less soil (better solutions).
        Affected by liquid viscosity and velocity multiplier, as well as soil resistance.
        """
        # Calculate distance to target
        distance = math.sqrt(
            sum((drop.position[i] - target_position[i]) ** 2 for i in range(self.dimension))
        )
        
        if distance < 1e-10:  # Avoid division by zero
            return

        # Calculate total soil content (weighted by resistance)
        total_soil_resistance = 0.0
        for soil_type in self.soil_types:
            soil_amount = drop.soil.get(soil_type, 0.0) + self.global_soil.get(soil_type, 0.0)
            resistance = soil_type.get_property("resistance")
            total_soil_resistance += soil_amount * resistance
        
        # Get liquid properties
        liquid_viscosity = drop.liquid_type.get_property("viscosity")
        liquid_velocity_mult = drop.liquid_type.get_property("velocity_multiplier")
        
        # Update velocity: higher velocity in low-soil areas
        # Velocity increases inversely with soil content, adjusted by viscosity
        velocity_factor = 1.0 / (1.0 + self.beta * total_soil_resistance)
        # Apply liquid properties
        velocity_factor *= liquid_velocity_mult / liquid_viscosity
        
        for i in range(self.dimension):
            direction = (target_position[i] - drop.position[i]) / distance
            # Update velocity component toward target
            drop.velocity[i] = (
                drop.velocity[i] * (1.0 - self.velocity_update_parameter) +
                direction * velocity_factor * self.initial_velocity * self.velocity_update_parameter
            )

    def _update_soil(self, drop: WaterDrop, new_fitness: float) -> None:
        """Update soil content based on fitness improvement and material properties.
        
        Drops pick up soil from worse positions and deposit it at better positions.
        Affected by liquid carrying capacity and soil pickup/deposition properties.
        """
        # Get liquid carrying capacity
        carrying_capacity = drop.liquid_type.get_property("carrying_capacity")
        fitness_change = abs(new_fitness - drop.fitness)
        
        # If fitness improved (decreased), pick up soil (less soil = better)
        if new_fitness < drop.fitness:
            # Pick up soil: better positions have less soil
            base_pickup = self.alpha / (1.0 + fitness_change)
            
            # Distribute pickup across soil types based on their properties
            for soil_type in self.soil_types:
                pickup_difficulty = soil_type.get_property("pickup_difficulty")
                # Easier to pick up = more pickup, adjusted by carrying capacity
                soil_pickup = base_pickup / (1.0 + pickup_difficulty) * carrying_capacity
                current_amount = drop.soil.get(soil_type, 0.0)
                drop.soil[soil_type] = current_amount + soil_pickup * self.soil_update_parameter
        else:
            # Deposit soil: worse positions accumulate more soil
            base_deposit = self.alpha * fitness_change
            
            # Distribute deposit across soil types
            for soil_type in self.soil_types:
                deposition_rate = soil_type.get_property("deposition_rate")
                # Higher deposition rate = more deposit
                soil_deposit = base_deposit * deposition_rate / carrying_capacity
                
                current_amount = drop.soil.get(soil_type, 0.0)
                deposit_amount = min(current_amount, soil_deposit * self.soil_update_parameter)
                drop.soil[soil_type] = max(0.0, current_amount - deposit_amount)
                self.global_soil[soil_type] = self.global_soil.get(soil_type, 0.0) + deposit_amount

    def _move(self, drop: WaterDrop) -> Vector:
        """Move the water drop based on its velocity and update position."""
        new_position = [drop.position[i] + drop.velocity[i] for i in range(self.dimension)]
        
        # Apply bounds constraints
        for i, (lo, hi) in enumerate(self.bounds):
            if new_position[i] < lo:
                new_position[i] = lo
                drop.velocity[i] *= -0.5  # Bounce back
            elif new_position[i] > hi:
                new_position[i] = hi
                drop.velocity[i] *= -0.5  # Bounce back
        
        return new_position

    def _select_target(self, drop: WaterDrop) -> Vector:
        """Select target position for the water drop.
        
        Combines attraction to best solution with exploration based on soil.
        """
        # Blend between best position and random exploration
        exploration_prob = 0.3  # 30% chance of exploration
        
        if self.random.random() < exploration_prob:
            # Explore: move toward a random position in the search space
            target = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
        else:
            # Exploit: move toward best solution
            if self.best:
                target = self.best.position[:]
            else:
                target = drop.position[:]
        
        return target

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Perform one optimization step."""
        if self.best is None:
            raise RuntimeError("Call initialise() before step().")

        for drop in self.population:
            # Select target position
            target = self._select_target(drop)
            
            # Update velocity toward target
            self._update_velocity(drop, target)
            
            # Move the drop
            new_position = self._move(drop)
            new_fitness = self.objective(new_position)
            
            # Update soil based on fitness change
            self._update_soil(drop, new_fitness)
            
            # Update drop position and fitness
            drop.position = new_position
            drop.fitness = new_fitness
            
            # Update best solution
            if drop.fitness < self.best.fitness:
                self.best = WaterDrop(
                    position=drop.position[:],
                    velocity=drop.velocity[:],
                    soil=drop.soil.copy(),
                    liquid_type=drop.liquid_type,
                    fitness=drop.fitness,
                )
        
        # Decay global soil over time (erosion/evaporation) - different rates per soil type
        for soil_type in self.soil_types:
            erosion_rate = soil_type.get_property("erosion_rate")
            self.global_soil[soil_type] *= (1.0 - 0.01 * erosion_rate)

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
            self.step()
        
        if self.best is None:
            raise RuntimeError("Algorithm did not initialise best solution")
        return self.best.position[:], self.best.fitness


def main() -> None:
    """Example usage of the IWD-CO algorithm."""
    print("=== IWD-CO with default settings (all material types) ===")
    optimizer = IWDCO(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        seed=42
    )
    best_position, best_value = optimizer.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])
    print()
    
    print("=== IWD-CO with specific liquid and soil types ===")
    optimizer2 = IWDCO(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        liquid_types=[LiquidType.OIL, LiquidType.ALCOHOL, LiquidType.HONEY],
        soil_types=[SoilType.SAND, SoilType.CLAY, SoilType.LOAM],
        liquid_distribution="uniform",
        seed=42
    )
    best_position2, best_value2 = optimizer2.run(iterations=200)
    print("Best value:", best_value2)
    print("Best position:", [round(x, 4) for x in best_position2])
    print(f"Liquid types used: {[lt.name for lt in optimizer2.liquid_types]}")
    print(f"Soil types used: {[st.name for st in optimizer2.soil_types]}")


if __name__ == "__main__":
    main()

