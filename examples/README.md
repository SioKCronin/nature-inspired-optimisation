# NIO Examples

This directory contains examples demonstrating various features of the nio library.

## Algorithm Comparison

Compare PPSO, IWD-CO, and Water Cycle Algorithm performance and explore hybrid approaches:

```bash
python examples/algorithm_comparison.py
```

This example:
- Runs all three algorithms on the same benchmark (Rastrigin function)
- Compares convergence rates, final solutions, and runtime
- Demonstrates a hybrid sequential approach combining all three algorithms
- Generates convergence plots (if matplotlib is available)

**Results typically show:**
- **PPSO**: Good exploration with proactive particles, slower but thorough
- **IWD-CO**: Fast convergence with water drop dynamics
- **Water Cycle Algorithm**: Excellent balance of exploration and exploitation
- **Hybrid Approach**: Sequential combination achieves best results by leveraging each algorithm's strengths

## Cultural Algorithm Visualization

Examples for visualizing the Cultural Algorithm optimization process.

## Quick Start

### Install Dependencies

```bash
pip install matplotlib numpy
```

For MP4 video output (optional):
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/)

### Run Examples

```bash
# Simple example
python examples/ca_visualization_example.py simple

# Custom configuration
python examples/ca_visualization_example.py custom

# Custom objective function
python examples/ca_visualization_example.py rosenbrock
```

Or use the command-line interface:

```bash
python -m nio.visualize_ca --iterations 50 --population-size 30 --output ca_optimization.mp4
```

## What the Visualization Shows

The animation displays:

- **Blue dots**: Population individuals exploring the search space
- **Red star**: Best individual found so far
- **Green dashed rectangle**: Normative bounds from belief space (adaptive search region)
- **Orange squares**: Situational knowledge (best examples stored in belief space)
- **Contour plot**: Objective function landscape (darker = better)
- **Text overlay**: Current iteration, best fitness, and bounds information

## Understanding the Process

1. **Initialization**: Population starts randomly distributed
2. **Acceptance**: Top individuals (based on accept_rate) influence belief space
3. **Belief Space Update**: 
   - Normative bounds shrink toward promising regions
   - Situational knowledge stores best examples
4. **Influence**: New individuals are generated influenced by cultural knowledge
5. **Evolution**: Population evolves through crossover, mutation, and cultural influence
6. **Convergence**: Population and belief space converge toward the optimum

## Customization

You can customize the visualization:

```python
from nio.visualize_ca import CAVisualizer
from nio import CulturalAlgorithm

ca = CulturalAlgorithm(
    bounds=((-5.12, 5.12), (-5.12, 5.12)),
    population_size=40,
    accept_rate=0.2,
    influence_rate=0.3,
)

visualizer = CAVisualizer(
    ca,
    save_path="my_animation.gif",
    fps=10,                    # Frames per second
    show_population=True,      # Show population individuals
    show_belief_space=True,    # Show normative bounds
    show_contour=True,         # Show objective contour
)

visualizer.create_animation(iterations=100)
```

