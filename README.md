# Nature-Inspired Optimisation

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
* Firefly Algorithm (FA) 
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

Install the package locally in editable mode::

```bash
pip install -e .
```

Once installed you can import `nio` from anywhere on your system.

### Typical Workflow

1. **Reduce large-scale multidimensional spaces using TDA** – apply your topological data analysis workflow first to simplify or extract the key structure of the search space.
2. **Define the optimization problem** – specify the objective function and bounds (often on the reduced representation).
3. **Run swarm optimization to find the global optimum** – choose any algorithm in `nio` (Bat, IWD-CO, Water Cycle, etc.) and run it for the desired number of iterations.
4. **Validate and interpret the best solution** – optionally verify the returned optimum with domain-specific checks or gradient-based refinements before deploying it.

### Using the Bat Algorithm

```python
from nio import BatAlgorithm

optimizer = BatAlgorithm(bounds=[(-5.12, 5.12)] * 5, population_size=40, seed=42)
best_position, best_value = optimizer.run(iterations=200)
print(best_value)
```

### Using the IWD-CO Algorithm

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

### Using the Water Cycle Algorithm

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

The algorithm supports 8 liquid types (FRESH_WATER, SALTWATER, DISTILLED_WATER, HOT_WATER, COLD_WATER, HEAVY_WATER, STEAM, ICE), each with different properties affecting flow speed, evaporation rate, and boiling point:

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
- **Flow speed**: Affects how fast water bodies move toward targets (STEAM is fastest, ICE is slowest)
- **Evaporation rate**: Affects probability of evaporation (HOT_WATER/STEAM evaporate faster, COLD_WATER/ICE slower)
- **Boiling point**: Affects distance threshold for evaporation (lower = evaporates at greater distances)
- **Density**: Affects flow behavior

Different liquid types create diverse optimization behaviors - fast-flowing liquids like STEAM explore quickly, while slower liquids like HEAVY_WATER provide more controlled convergence.

### Command-line demo

```bash
python -m nio --iterations 200 --dimension 5
```

This runs the reference implementation from Yang (2010) on a Rastrigin benchmark.

### Visualizing the Cultural Algorithm

Create animated visualizations showing the optimization process:

**Interactive HTML Visualization (opens in browser):**

```python
from nio.visualize_ca import visualize_ca_html

html_path = visualize_ca_html(
    bounds=((-5.12, 5.12), (-5.12, 5.12)),
    population_size=30,
    iterations=50,
    save_path="ca_visualization.html",
    seed=42
)
# Opens in browser with interactive controls (play, pause, step, slider)
```

**Video/GIF Visualization:**

```bash
# Install visualization dependencies
pip install matplotlib numpy

# Create visualization
python -m nio.visualize_ca --iterations 50 --population-size 30 --output ca_optimization.mp4
```

Or use in Python:

```python
from nio.visualize_ca import visualize_ca

visualize_ca(
    bounds=((-5.12, 5.12), (-5.12, 5.12)),
    population_size=30,
    iterations=50,
    save_path="ca_optimization.mp4",
    seed=42
)
```

The visualization shows:
- Population individuals (blue dots)
- Best individual (red star)
- Normative bounds from belief space (green rectangle)
- Situational knowledge (orange squares)
- Objective function contour

The HTML version includes interactive controls: play/pause, step-by-step navigation, and a slider to jump to any iteration.

See `examples/README.md` for more visualization examples.
