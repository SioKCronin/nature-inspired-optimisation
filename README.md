# Nature-Inspired Optimisation

My goal with this project is to celebrate optimization strategies from our blue planet. I see our collective 
documentation of strategies from nature as a heritage we can all observe, document, and celebrate together. 
My hope is we can use these algorithms as a meeting ground for refining our collective understanding.

Links to original papers introducing (or meta-analysis overviews of) the following algorithms/heuristics/methods:

* Genetic Algorithms (GA)
* Particle Swarm Optimization (PSO)
* Artificial immune systems (AIS) 
* Boids
* Memetic Algorithm (MA)
* Ant Colony Optimization (ACO)
* [Cultural Algorithms (CA)](https://link.springer.com/book/10.1007/978-981-19-4633-2)
* Particle Swarm Optimization (PSO)
* Self-propelled Particles
* Differential Evolution (DE)
* Bacterial Foraging Optimization
* Marriage in Honey Bees (MHB) 
* Artificial Fish School
* [Bacteria Chemotaxis (BC)](https://ieeexplore.ieee.org/document/985689)
* [Social Cognitive Optimization (SCO)](https://ieeexplore.ieee.org/document/5660738)
* Artificial Bee Colony
* Glowworm Swarm Optimization (GSO)
* Honey-Bees Mating Optimization (HBMO)
* Invasive Weed Optimization (IWO)
* Shuffled Frog Leaping Algorithm (SFLA)
* [Intelligent Water Drops - Continuous Optimization(IWD-CO)](https://www.sciencedirect.com/science/article/pii/S1877042812000341)
* [River Formation Dynamics](https://www.sciencedirect.com/science/article/abs/pii/S1877750317307184)
* Biogeography-based Optimization (BBO)
* Roach Infestation Optimization (RIO)
* Bacterial Evolutionary Algorithm (BEA)
* Cuckoo Search (CS)
* [Firefly Algorithm (FA)](https://arxiv.org/abs/1003.1466) 
* Gravitational Search Algorithm (GSA)
* [Bat Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S1877750322002903)
* [Phillippine Eagle Optimization Algorithm](https://ieeexplore.ieee.org/document/9732449)
* Fireworks algorithm
* [Altruistic Population Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S037847542300109X)
* Spiral Dynamic Algorithm (SDA)
* Strawberry Algorithm
* Artificial Algae Algorithm (AAA) 
* Bacterial Colony Optimization
* Flower pollination algorithm (FPA)
* Krill Herd
* Water Cycle Algorithm 
* [Proactive Particle Swarm Optimization (PPSO)](https://ieeexplore.ieee.org/document/7337957)
* [Dragonfly Algorithm (DA)](https://link.springer.com/article/10.1007/s00521-015-1920-1)
* Black Holes Algorithm
* Cuttlefish Algorithm
* Gases Brownian Motion Optimization
* Mine blast algorithm
* Plant Propagation Algorithm
* Social Spider Optimization (SSO)
* Spider Monkey Optimization (SMO) 
* Animal Migration Optimization (AMO) 
* Artificial Ecosystem Algorithm (AEA)
* Bird Mating Optimizer
* [Forest Optimization Algorithm (FOA)](https://www.sciencedirect.com/science/article/abs/pii/S0957417414002899)
* Grey Wolf Optimizer
* Lion Optimization Algorithm (LOA)
* Optics Inspired Optimization (OIO)
* The Raven Roosting Optimisation Algorithm
* [Water Wave Optimization](https://www.sciencedirect.com/science/article/pii/S0305054814002652)
* Collective animal behavior (CAB)
* Aritificial Chemical Process Algorithm
* Bull optimization algorithm
* Elephent herding optimization (EHO)

# Publications

* [Algorithms](http://www.mdpi.com/journal/algorithms)
* [Journal of Algorithms](https://www.sciencedirect.com/journal/journal-of-algorithms)
* [Swarm and Evolutionary Computation](https://www.journals.elsevier.com/swarm-and-evolutionary-computation/)
* [International Journal of Swarm Intelligence and Evolutionary Computation](https://www.omicsonline.org/swarm-intelligence-evolutionary-computation.php#)
* [Swarm Intelligence](https://link.springer.com/journal/11721)
* [Evolutionary Intelligence](http://www.springer.com/engineering/computational+intelligence+and+complexity/journal/12065)

# Conferences

* [GECCO](http://gecco-2018.sigevo.org/index.html/tiki-index.php?page=HomePage)

# Research teams

* [Tübingen](http://www.ra.cs.uni-tuebingen.de/links/genetisch/welcome_e.html)


## Getting Started

Install the package locally. Once installed you can import `nio` from anywhere on your system.


### EXAMPLE: Using the IWD-CO Algorithm

The Intelligent Water Drops - Continuous Optimization (IWD-CO) algorithm simulates water drops flowing through a landscape, where drops move toward areas with less soil (better solutions). The implementation includes multiple liquid and soil types, each with distinct physical properties that affect optimization behavior.

**Basic usage:**

```python
from nio import IWDCO

optimizer = IWDCO(bounds=[(-5.12, 5.12)] * 5, population_size=40, seed=42)
best_position, best_value = optimizer.run(iterations=200)
print(best_value)
```

**Using specific material types:**

The algorithm supports 6 liquid types (H2O, OIL, ALCOHOL, GLYCEROL, MERCURY, HONEY) and 6 soil types (SAND, CLAY, SILT, GRAVEL, LOAM, PEAT), each with different properties affecting velocity, soil pickup, and deposition:

```python
from nio import IWDCO, LiquidType, SoilType

# Use specific liquid and soil types
optimizer = IWDCO(
    bounds=[(-5.12, 5.12)] * 5,
    population_size=40,
    liquid_types=[LiquidType.OIL, LiquidType.ALCOHOL, LiquidType.HONEY],
    soil_types=[SoilType.SAND, SoilType.CLAY, SoilType.LOAM],
    liquid_distribution="uniform",  # or "random"
    seed=42
)
best_position, best_value = optimizer.run(iterations=200)
```

**Material properties:**
- **Liquid types** affect velocity (viscosity, velocity multiplier) and soil carrying capacity
- **Soil types** affect movement resistance, pickup difficulty, deposition rate, and erosion rate
- Different combinations create diverse optimization behaviors

### EXAMPLE: Using the Water Cycle Algorithm

The Water Cycle Algorithm (WCA) simulates the natural water cycle process, where streams flow toward rivers, rivers flow toward the sea, and evaporation/raining processes provide exploration.

**Basic usage:**

```python
from nio import WaterCycleAlgorithm

optimizer = WaterCycleAlgorithm(
    bounds=[(-5.12, 5.12)] * 5,
    population_size=40,
    num_rivers=4,
    seed=42
)
best_position, best_value = optimizer.run(iterations=200)
```

**Algorithm features:**
- **Sea**: Best solution (global best)
- **Rivers**: Better solutions that streams flow toward
- **Streams**: Population members that flow toward rivers or sea
- **Flow process**: Streams and rivers move toward better solutions
- **Evaporation**: When streams/rivers get close to sea, they evaporate
- **Raining**: Evaporated water creates new random solutions for exploration

**Parameters:**
- `num_rivers`: Number of rivers (better solutions), typically 3-5
- `evaporation_rate`: Base probability of evaporation (controls exploration)
- `max_evaporation_distance`: Maximum distance for evaporation to occur
- `flow_rate`: Base rate at which water bodies move toward targets

**Using specific liquid types:**

The algorithm supports 7 liquid types (FRESH_WATER, SALTWATER, DISTILLED_WATER, HOT_WATER, COLD_WATER, HEAVY_WATER, STEAM), each with different properties affecting flow speed, evaporation rate, and boiling point:

```python
from nio import WaterCycleAlgorithm, WCA_LiquidType

# Use specific liquid types
optimizer = WaterCycleAlgorithm(
    bounds=[(-5.12, 5.12)] * 5,
    population_size=40,
    num_rivers=4,
    liquid_types=[WCA_LiquidType.HOT_WATER, WCA_LiquidType.STEAM, WCA_LiquidType.FRESH_WATER],
    liquid_distribution="uniform",  # or "random"
    seed=42
)
best_position, best_value = optimizer.run(iterations=200)
```

**Liquid properties:**
- **Flow speed**: Affects how fast water bodies move toward targets (STEAM is fastest, HEAVY_WATER is slowest)
- **Evaporation rate**: Affects probability of evaporation (HOT_WATER/STEAM evaporate faster, COLD_WATER slower)
- **Boiling point**: Affects distance threshold for evaporation (lower = evaporates at greater distances)
- **Density**: Affects flow behavior

Different liquid types create diverse optimization behaviors - fast-flowing liquids like STEAM explore quickly, while slower liquids like HEAVY_WATER provide more controlled convergence.


## Contributing

Contributions are welcome across algorithms, benchmarks, documentation, and examples.

- Open an issue first for substantial feature work to align on scope.
- Fork the repo and create a focused branch per change.
- Add or update tests and examples where practical.
- Keep algorithm references (paper links/citations) in the README or module docstrings.
- Open a pull request with a clear summary of motivation, approach, and validation steps.

## License

This project is licensed under the MIT License.

Copyright (c) 2018-present Nature-Inspired Optimisation contributors.
