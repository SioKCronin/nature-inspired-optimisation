"""
Water Cycle Algorithm (WCA) Implementation

Reference:
Eskandar, H., Sadollah, A., Bahreininejad, A., & Hamdi, M. (2012).
Water cycle algorithm–A novel metaheuristic optimization method for solving constrained 
engineering optimization problems. Computers & Structures, 110, 151-166.

The algorithm simulates the water cycle process:
- Streams flow toward rivers (better solutions)
- Rivers flow toward the sea (best solution)
- Evaporation occurs when streams/rivers get close to the sea
- Rain (new solutions) falls in random locations
- The cycle continues with flow and evaporation processes

Enhanced with multiple liquid types, each with distinct properties affecting
flow speed, evaporation rate, and behavior in the water cycle.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Sequence, Tuple

Vector = List[float]
Objective = Callable[[Vector], float]


def rastrigin(position: Sequence[float]) -> float:
    """Default benchmark function (global minimum at 0)."""
    return 10 * len(position) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in position)


class LiquidType(Enum):
    """Types of liquids with different properties affecting water cycle behavior."""
    FRESH_WATER = {
        "flow_speed": 1.0,        # Baseline flow speed
        "evaporation_rate": 1.0,   # Baseline evaporation rate
        "density": 1.0,            # Baseline density
        "boiling_point": 1.0,      # Baseline boiling point (affects evaporation threshold)
    }
    SALTWATER = {
        "flow_speed": 0.9,         # Slightly slower due to higher density
        "evaporation_rate": 0.8,   # Slower evaporation (salt lowers vapor pressure)
        "density": 1.03,            # Higher density
        "boiling_point": 1.02,      # Slightly higher boiling point
    }
    DISTILLED_WATER = {
        "flow_speed": 1.0,         # Same as fresh water
        "evaporation_rate": 1.2,    # Faster evaporation (pure, no impurities)
        "density": 1.0,             # Same density
        "boiling_point": 0.98,      # Slightly lower boiling point
    }
    HOT_WATER = {
        "flow_speed": 1.1,         # Faster flow (lower viscosity when hot)
        "evaporation_rate": 1.5,    # Much faster evaporation
        "density": 0.98,            # Lower density (thermal expansion)
        "boiling_point": 0.9,       # Lower effective boiling point
    }
    COLD_WATER = {
        "flow_speed": 0.85,        # Slower flow (higher viscosity when cold)
        "evaporation_rate": 0.5,    # Much slower evaporation
        "density": 1.0,             # Same density
        "boiling_point": 1.1,       # Higher effective boiling point
    }
    HEAVY_WATER = {
        "flow_speed": 0.7,         # Much slower flow (higher density)
        "evaporation_rate": 0.6,    # Slower evaporation
        "density": 1.1,             # Higher density
        "boiling_point": 1.05,       # Higher boiling point
    }
    STEAM = {
        "flow_speed": 2.0,         # Very fast (gas phase)
        "evaporation_rate": 3.0,    # Already evaporated, very high rate
        "density": 0.6,             # Much lower density
        "boiling_point": 0.5,       # Very low (already past boiling)
    }
    ICE = {
        "flow_speed": 0.1,         # Very slow (solid phase)
        "evaporation_rate": 0.1,    # Very slow (sublimation)
        "density": 0.92,            # Lower density (ice floats)
        "boiling_point": 2.0,       # Very high (needs to melt first)
    }

    def get_property(self, prop: str) -> float:
        """Get a property value for this liquid type."""
        return self.value[prop]


@dataclass
class WaterBody:
    """Represents a water body (stream, river, or sea) in the WCA."""
    position: Vector
    fitness: float
    liquid_type: LiquidType = LiquidType.FRESH_WATER


class WaterCycleAlgorithm:
    """Water Cycle Algorithm for continuous optimization.
    
    Enhanced with multiple liquid types, each affecting flow and evaporation behavior.
    """

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        num_rivers: int = 4,
        evaporation_rate: float = 0.1,
        max_evaporation_distance: float = 0.1,
        flow_rate: float = 2.0,
        liquid_types: Sequence[LiquidType] | None = None,  # Available liquid types
        liquid_distribution: str = "uniform",  # "uniform" or "random"
        seed: int | None = None,
    ) -> None:
        """Initialize the Water Cycle Algorithm.

        Args:
            objective: Objective function to minimize
            bounds: Search space boundaries for each dimension
            population_size: Total number of water bodies (streams + rivers + 1 sea)
            num_rivers: Number of rivers (better solutions)
            evaporation_rate: Base rate of evaporation (controls exploration)
            max_evaporation_distance: Maximum distance for evaporation to occur
            flow_rate: Base rate at which streams/rivers flow toward better solutions
            liquid_types: Available liquid types (default: all)
            liquid_distribution: How to assign liquid types ("uniform" or "random")
            seed: Random seed for reproducibility
        """
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if num_rivers < 0 or num_rivers >= population_size - 1:
            raise ValueError("num_rivers must be between 0 and population_size - 1")
        
        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.num_rivers = num_rivers
        self.num_streams = population_size - num_rivers - 1  # -1 for the sea
        self.base_evaporation_rate = evaporation_rate
        self.max_evaporation_distance = max_evaporation_distance
        self.base_flow_rate = flow_rate
        self.random = random.Random(seed)

        # Material type configuration
        self.liquid_types = liquid_types if liquid_types else list(LiquidType)
        self.liquid_distribution = liquid_distribution

        # Water bodies: sea (best), rivers (better), streams (rest)
        self.sea: WaterBody | None = None
        self.rivers: List[WaterBody] = []
        self.streams: List[WaterBody] = []
        
        # Assignment: which stream flows to which river/sea
        self.stream_assignments: List[int] = []  # Index of river/sea for each stream

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        """Initialize the population of water bodies."""
        # Create all water bodies
        all_bodies: List[WaterBody] = []
        for i in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            fitness = self.objective(position)
            
            # Assign liquid type
            if self.liquid_distribution == "uniform":
                liquid_type = self.liquid_types[i % len(self.liquid_types)]
            else:  # random
                liquid_type = self.random.choice(self.liquid_types)
            
            all_bodies.append(WaterBody(position, fitness, liquid_type))

        # Sort by fitness (best first)
        all_bodies.sort(key=lambda w: w.fitness)

        # Assign: sea (best), rivers (next best), streams (rest)
        self.sea = all_bodies[0]
        self.rivers = all_bodies[1 : 1 + self.num_rivers]
        self.streams = all_bodies[1 + self.num_rivers :]

        # Assign streams to rivers/sea based on fitness (roulette wheel selection)
        self._assign_streams()

    def _assign_streams(self) -> None:
        """Assign streams to rivers or sea using roulette wheel selection."""
        # Combine sea and rivers for assignment
        targets = [self.sea] + self.rivers
        target_fitnesses = [t.fitness for t in targets]
        
        # Convert to probabilities (inverse fitness - better fitness = higher probability)
        # Use negative fitness to handle minimization
        max_fitness = max(target_fitnesses)
        probabilities = [max_fitness - f + 1e-10 for f in target_fitnesses]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # Cumulative probabilities
        cum_probs = []
        cum_sum = 0.0
        for p in probabilities:
            cum_sum += p
            cum_probs.append(cum_sum)
        
        # Assign each stream
        self.stream_assignments = []
        for _ in self.streams:
            r = self.random.random()
            # Find which target this stream belongs to
            target_idx = 0
            for i, cum_prob in enumerate(cum_probs):
                if r <= cum_prob:
                    target_idx = i
                    break
            self.stream_assignments.append(target_idx)

    # ------------------------------------------------------------------
    def _flow_toward_target(self, water: WaterBody, target: WaterBody) -> Vector:
        """Move water body toward target based on flow rate and liquid properties."""
        # Get liquid flow speed
        flow_speed = water.liquid_type.get_property("flow_speed")
        effective_flow_rate = self.base_flow_rate * flow_speed
        
        new_position = []
        for i in range(self.dimension):
            # Flow toward target
            direction = target.position[i] - water.position[i]
            step = self.random.uniform(0, effective_flow_rate) * direction
            new_pos = water.position[i] + step
            
            # Apply bounds
            lo, hi = self.bounds[i]
            new_pos = max(lo, min(hi, new_pos))
            new_position.append(new_pos)
        
        return new_position

    def _evaporation_condition(self, water: WaterBody, target: WaterBody) -> bool:
        """Check if evaporation should occur (water is close to target).
        
        Boiling point affects the distance threshold - liquids with lower
        boiling points evaporate at greater distances.
        """
        distance = math.sqrt(
            sum((water.position[i] - target.position[i]) ** 2 
                for i in range(self.dimension))
        )
        
        # Adjust evaporation distance based on boiling point
        boiling_point = water.liquid_type.get_property("boiling_point")
        # Lower boiling point = evaporates at greater distance
        effective_distance = self.max_evaporation_distance / boiling_point
        
        return distance < effective_distance

    def _get_evaporation_rate(self, water: WaterBody) -> float:
        """Get effective evaporation rate for a water body based on liquid type."""
        liquid_evap_rate = water.liquid_type.get_property("evaporation_rate")
        return self.base_evaporation_rate * liquid_evap_rate

    def _rain(self, num_drops: int) -> List[WaterBody]:
        """Generate new water bodies through rain (random positions).
        
        Rain can have different liquid types based on distribution.
        """
        new_bodies = []
        for _ in range(num_drops):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            fitness = self.objective(position)
            
            # Assign liquid type to rain (always random for rain)
            liquid_type = self.random.choice(self.liquid_types)
            
            new_bodies.append(WaterBody(position, fitness, liquid_type))
        return new_bodies

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Perform one optimization step."""
        if self.sea is None:
            raise RuntimeError("Call initialise() before step().")

        # Phase 1: Flow process
        # Streams flow toward their assigned rivers/sea
        for i, stream in enumerate(self.streams):
            target_idx = self.stream_assignments[i]
            if target_idx == 0:
                target = self.sea
            else:
                target = self.rivers[target_idx - 1]
            
            # Flow toward target
            new_position = self._flow_toward_target(stream, target)
            new_fitness = self.objective(new_position)
            
            # Update if better
            if new_fitness < stream.fitness:
                stream.position = new_position
                stream.fitness = new_fitness
            
            # Check if stream is better than its target river
            if target_idx > 0:  # Not sea
                river = self.rivers[target_idx - 1]
                if stream.fitness < river.fitness:
                    # Swap: stream becomes river, river becomes stream
                    stream.position, river.position = river.position, stream.position
                    stream.fitness, river.fitness = river.fitness, stream.fitness

        # Rivers flow toward sea
        for river in self.rivers:
            new_position = self._flow_toward_target(river, self.sea)
            new_fitness = self.objective(new_position)
            
            # Update if better
            if new_fitness < river.fitness:
                river.position = new_position
                river.fitness = new_fitness
            
            # Check if river is better than sea
            if river.fitness < self.sea.fitness:
                # Swap: river becomes sea, sea becomes river
                river.position, self.sea.position = self.sea.position, river.position
                river.fitness, self.sea.fitness = self.sea.fitness, river.fitness

        # Phase 2: Evaporation and raining
        evaporated_streams = []
        evaporated_rivers = []
        
        # Check streams for evaporation
        for i, stream in enumerate(self.streams):
            target_idx = self.stream_assignments[i]
            if target_idx == 0:
                target = self.sea
            else:
                target = self.rivers[target_idx - 1]
            
            if self._evaporation_condition(stream, target):
                # Use liquid-specific evaporation rate
                evap_rate = self._get_evaporation_rate(stream)
                if self.random.random() < evap_rate:
                    evaporated_streams.append(i)
        
        # Check rivers for evaporation
        for i, river in enumerate(self.rivers):
            if self._evaporation_condition(river, self.sea):
                # Use liquid-specific evaporation rate
                evap_rate = self._get_evaporation_rate(river)
                if self.random.random() < evap_rate:
                    evaporated_rivers.append(i)
        
        # Replace evaporated streams with rain
        for idx in reversed(evaporated_streams):  # Reverse to maintain indices
            new_bodies = self._rain(1)
            self.streams[idx] = new_bodies[0]
            # Reassign stream
            self._assign_streams()
        
        # Replace evaporated rivers with rain
        for idx in reversed(evaporated_rivers):
            new_bodies = self._rain(1)
            self.rivers[idx] = new_bodies[0]
        
        # Reassign all streams after evaporation
        self._assign_streams()

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
        
        if self.sea is None:
            raise RuntimeError("Algorithm did not initialise sea solution")
        return self.sea.position[:], self.sea.fitness


def main() -> None:
    """Example usage of the Water Cycle Algorithm."""
    optimizer = WaterCycleAlgorithm(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        num_rivers=4,
        seed=42
    )
    best_position, best_value = optimizer.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])


if __name__ == "__main__":
    main()

