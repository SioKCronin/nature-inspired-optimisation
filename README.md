# Nature-Inspired Optimisation

My goal with this project is to celebrate optimization strategies from our blue planet. I see our collective 
documentation of strategies from nature as a heritage we can all observe, document, and celebrate together. 
My hope is we can use these algorithms as a meeting ground for refining our collective understanding.

Every method below is implemented. Continuous optimizers share one contract: construct with an objective, bounds, population size, and seed, then call `run(iterations)` to minimise. Short names live in `nio.OPTIMIZERS`. Boids and self-propelled particles are movement models. Ant colony and river formation dynamics also run as grid pathfinders.

Links to original papers introducing (or meta-analysis overviews of) the following algorithms/heuristics/methods:

* Genetic Algorithms (GA) — `GeneticAlgorithm`
* Particle Swarm Optimization (PSO) — `ParticleSwarmOptimization`
* Artificial immune systems (AIS) — `ClonalSelection`
* Boids — `boids_step`
* Memetic Algorithm (MA) — `MemeticAlgorithm`
* Ant Colony Optimization (ACO) — `AntColonyOptimization`, and `AntColonyPath` on a grid
* [Cultural Algorithms (CA)](https://link.springer.com/book/10.1007/978-981-19-4633-2) — `CulturalAlgorithm`
* Self-propelled Particles — `vicsek_step`
* Differential Evolution (DE) — `DifferentialEvolution`
* Bacterial Foraging Optimization — `BacterialForagingOptimization`
* Marriage in Honey Bees (MHB) — `MarriageInHoneyBees`
* Artificial Fish School — `ArtificialFishSchool`
* [Bacteria Chemotaxis (BC)](https://ieeexplore.ieee.org/document/985689) — `BacterialChemotaxis`
* [Social Cognitive Optimization (SCO)](https://ieeexplore.ieee.org/document/5660738) — `SocialCognitiveOptimization`
* Artificial Bee Colony — `ArtificialBeeColony`
* Glowworm Swarm Optimization (GSO) — `GlowwormSwarmOptimization`
* Honey-Bees Mating Optimization (HBMO) — `HoneyBeeMatingOptimization`
* Invasive Weed Optimization (IWO) — `InvasiveWeedOptimization`
* Shuffled Frog Leaping Algorithm (SFLA) — `ShuffledFrogLeaping`
* [Intelligent Water Drops - Continuous Optimization(IWD-CO)](https://www.sciencedirect.com/science/article/pii/S1877042812000341) — `IWDCO`
* [River Formation Dynamics](https://www.sciencedirect.com/science/article/abs/pii/S1877750317307184) — `RiverFormationDynamics`
* Biogeography-based Optimization (BBO) — `BiogeographyBasedOptimization`
* Roach Infestation Optimization (RIO) — `RoachInfestationOptimization`
* Bacterial Evolutionary Algorithm (BEA) — `BacterialEvolutionaryAlgorithm`
* Cuckoo Search (CS) — `CuckooSearch`
* [Firefly Algorithm (FA)](https://arxiv.org/abs/1003.1466) — `FireflyAlgorithm`
* Gravitational Search Algorithm (GSA) — `GravitationalSearchAlgorithm`
* [Bat Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S1877750322002903) — `BatAlgorithm`
* [Phillippine Eagle Optimization Algorithm](https://ieeexplore.ieee.org/document/9732449) — `PhilippineEagleOptimization`
* Fireworks algorithm — `FireworksAlgorithm`
* [Altruistic Population Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S037847542300109X) — `AltruisticPopulationAlgorithm`
* Spiral Dynamic Algorithm (SDA) — `SpiralDynamicAlgorithm`
* Strawberry Algorithm — `StrawberryAlgorithm`
* Artificial Algae Algorithm (AAA) — `ArtificialAlgaeAlgorithm`
* Bacterial Colony Optimization — `BacterialColonyOptimization`
* Flower pollination algorithm (FPA) — `FlowerPollinationAlgorithm`
* Krill Herd — `KrillHerd`
* Water Cycle Algorithm — `WaterCycleAlgorithm`
* [Proactive Particle Swarm Optimization (PPSO)](https://ieeexplore.ieee.org/document/7337957) — `PPSO`
* [Dragonfly Algorithm (DA)](https://link.springer.com/article/10.1007/s00521-015-1920-1) — `DragonflyAlgorithm`
* Black Holes Algorithm — `BlackHoleAlgorithm`
* Cuttlefish Algorithm — `CuttlefishAlgorithm`
* Gases Brownian Motion Optimization — `GasesBrownianMotionOptimization`
* Mine blast algorithm — `MineBlastAlgorithm`
* Plant Propagation Algorithm — `PlantPropagationAlgorithm`
* Social Spider Optimization (SSO) — `SocialSpiderOptimization`
* Spider Monkey Optimization (SMO) — `SpiderMonkeyOptimization`
* Animal Migration Optimization (AMO) — `AnimalMigrationOptimization`
* Artificial Ecosystem Algorithm (AEA) — `ArtificialEcosystemAlgorithm`
* Bird Mating Optimizer — `BirdMatingOptimizer`
* [Forest Optimization Algorithm (FOA)](https://www.sciencedirect.com/science/article/abs/pii/S0957417414002899) — `ForestOptimizationAlgorithm`
* Grey Wolf Optimizer — `GreyWolfOptimizer`
* Lion Optimization Algorithm (LOA) — `LionOptimizationAlgorithm`
* Optics Inspired Optimization (OIO) — `OpticsInspiredOptimization`
* The Raven Roosting Optimisation Algorithm — `RavenRoostingOptimization`
* [Water Wave Optimization](https://www.sciencedirect.com/science/article/pii/S0305054814002652) — `WaterWaveOptimization`
* Collective animal behavior (CAB) — `CollectiveAnimalBehavior`
* Aritificial Chemical Process Algorithm — `ArtificialChemicalReactionOptimization`
* Bull optimization algorithm — `BullOptimizationAlgorithm`
* Elephent herding optimization (EHO) — `ElephantHerdingOptimization`

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


## Lead drones

The corpus is the heritage. The flight environment is what a lead drone can actually run.

`LeaderSwarmEnv` is a small horizontal world: a handful of drones, a couple of them designated as leaders, circular obstacles, and one shared reward. Followers hold formation with boids. Leaders choose a compass heading and a speed. Nothing here depends on Gymnasium or a neural-network library. `reset` and `step` return observation, reward, terminated, truncated, and info, so another agent can drive the same world.

```python
from nio.flight import LeaderSwarmEnv, plan_actions, train_policy

env = LeaderSwarmEnv(n_drones=8, n_leaders=2, seed=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(plan_actions(env))
```

Two ways to choose that step:

- `plan_actions` turns the next waypoint into a cost — distance to the goal, clearance from obstacles, leaders staying together — and minimises it with any name in `nio.OPTIMIZERS`. Grey wolf is the default. `plan_actions_grid` rasterises the same world and follows an ant-colony or river-formation path.
- `train_policy` fits a heading from two dot products with the leader's observation. On the default world that is 28 floats. The weights start pointed at the goal, and training keeps the episodes that beat the running mean. `to_json` / `from_json` is the form you would copy onto the aircraft. Inference is those two dot products.

```bash
python examples/leader_swarm_demo.py
```

The demo flies the same world three ways. A fixed eastward heading scrapes the obstacle. The grey-wolf plan and the trained heading both reach the goal. If matplotlib is installed, the trails are written to `examples/leader_swarm.png`.

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
