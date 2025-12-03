#!/usr/bin/env python3
"""
Algorithm Comparison: PPSO, IWD-CO, and Water Cycle Algorithm

This example demonstrates how PPSO, IWD-CO, and Water Cycle Algorithm
perform on the same benchmark problem, and shows how they can be combined
in a hybrid approach.

Requirements:
    pip install matplotlib numpy
"""

import time
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from nio import PPSO, IWDCO, WaterCycleAlgorithm, rastrigin


def run_algorithm_comparison(
    bounds: List[Tuple[float, float]],
    population_size: int = 40,
    iterations: int = 200,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Compare PPSO, IWD-CO, and Water Cycle Algorithm on the same problem."""
    
    results = {
        'ppso': {'best_values': [], 'best_positions': [], 'runtime': 0, 'final_best': float('inf')},
        'iwd_co': {'best_values': [], 'best_positions': [], 'runtime': 0, 'final_best': float('inf')},
        'water_cycle': {'best_values': [], 'best_positions': [], 'runtime': 0, 'final_best': float('inf')},
    }
    
    # Run PPSO
    if verbose:
        print("Running PPSO...")
    start = time.time()
    ppso = PPSO(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        proactive_ratio=0.25,
        seed=seed
    )
    ppso.initialise()
    
    for i in range(iterations):
        ppso.step(i, iterations)
        if ppso.best:
            results['ppso']['best_values'].append(ppso.best.best_fitness)
            results['ppso']['best_positions'].append(ppso.best.best_position[:])
    results['ppso']['runtime'] = time.time() - start
    results['ppso']['final_best'] = ppso.best.best_fitness if ppso.best else float('inf')
    
    if verbose:
        print(f"  Final best: {results['ppso']['final_best']:.6f}")
        print(f"  Runtime: {results['ppso']['runtime']:.3f}s")
    
    # Run IWD-CO
    if verbose:
        print("\nRunning IWD-CO...")
    start = time.time()
    iwd_co = IWDCO(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        seed=seed
    )
    iwd_co.initialise()
    
    for i in range(iterations):
        iwd_co.step()
        if iwd_co.best:
            results['iwd_co']['best_values'].append(iwd_co.best.fitness)
            results['iwd_co']['best_positions'].append(iwd_co.best.position[:])
    results['iwd_co']['runtime'] = time.time() - start
    results['iwd_co']['final_best'] = iwd_co.best.fitness if iwd_co.best else float('inf')
    
    if verbose:
        print(f"  Final best: {results['iwd_co']['final_best']:.6f}")
        print(f"  Runtime: {results['iwd_co']['runtime']:.3f}s")
    
    # Run Water Cycle Algorithm
    if verbose:
        print("\nRunning Water Cycle Algorithm...")
    start = time.time()
    wca = WaterCycleAlgorithm(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        num_rivers=4,
        seed=seed
    )
    wca.initialise()
    
    for i in range(iterations):
        wca.step()
        if wca.sea:
            results['water_cycle']['best_values'].append(wca.sea.fitness)
            results['water_cycle']['best_positions'].append(wca.sea.position[:])
    results['water_cycle']['runtime'] = time.time() - start
    results['water_cycle']['final_best'] = wca.sea.fitness if wca.sea else float('inf')
    
    if verbose:
        print(f"  Final best: {results['water_cycle']['final_best']:.6f}")
        print(f"  Runtime: {results['water_cycle']['runtime']:.3f}s")
    
    return results


def plot_convergence_comparison(results: Dict, save_path: str = "algorithm_convergence.png"):
    """Plot convergence curves for all three algorithms."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available, skipping plot generation")
        print("  Install with: pip install matplotlib")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(results['ppso']['best_values']))
    
    ax.plot(iterations, results['ppso']['best_values'], 
            label=f'PPSO (final: {results["ppso"]["final_best"]:.4f})', 
            linewidth=2, alpha=0.8)
    ax.plot(iterations, results['iwd_co']['best_values'], 
            label=f'IWD-CO (final: {results["iwd_co"]["final_best"]:.4f})', 
            linewidth=2, alpha=0.8)
    ax.plot(iterations, results['water_cycle']['best_values'], 
            label=f'Water Cycle (final: {results["water_cycle"]["final_best"]:.4f})', 
            linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness Value', fontsize=12)
    ax.set_title('Algorithm Convergence Comparison (Rastrigin Function)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Convergence plot saved to {save_path}")
    return fig


def print_comparison_summary(results: Dict):
    """Print a summary comparison of the algorithms."""
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 60)
    
    # Find best algorithm
    best_algo = min(results.keys(), key=lambda k: results[k]['final_best'])
    
    for algo_name in ['ppso', 'iwd_co', 'water_cycle']:
        algo_display = {
            'ppso': 'PPSO',
            'iwd_co': 'IWD-CO',
            'water_cycle': 'Water Cycle Algorithm'
        }[algo_name]
        
        result = results[algo_name]
        marker = " 🏆" if algo_name == best_algo else ""
        
        print(f"\n{algo_display}:{marker}")
        print(f"  Final Best Value: {result['final_best']:.6f}")
        print(f"  Runtime: {result['runtime']:.3f}s")
        print(f"  Improvement: {result['best_values'][0] - result['final_best']:.6f}")
        print(f"  Convergence Rate: {(result['best_values'][0] - result['final_best']) / result['runtime']:.2f} units/sec")
    
    print("\n" + "=" * 60)


def hybrid_approach_example(
    bounds: List[Tuple[float, float]],
    population_size: int = 40,
    iterations: int = 200,
    seed: int = 42
):
    """
    Demonstrate a hybrid approach: use different algorithms in sequence.
    
    Strategy:
    1. Start with PPSO for exploration (proactive particles find unexplored regions)
    2. Switch to IWD-CO for refinement (water drops flow toward better solutions)
    3. Finish with Water Cycle for fine-tuning (evaporation/raining provides local search)
    """
    print("\n" + "=" * 60)
    print("HYBRID APPROACH: Sequential Algorithm Combination")
    print("=" * 60)
    
    # Phase 1: PPSO for exploration
    print("\nPhase 1: PPSO (Exploration)...")
    ppso = PPSO(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        proactive_ratio=0.3,  # Higher proactive ratio for more exploration
        seed=seed
    )
    ppso.initialise()
    
    phase1_iterations = iterations // 3
    for i in range(phase1_iterations):
        ppso.step(i, phase1_iterations)
    
    best_after_ppso = ppso.best.best_fitness if ppso.best else float('inf')
    print(f"  Best after PPSO: {best_after_ppso:.6f}")
    
    # Phase 2: IWD-CO for refinement
    print("\nPhase 2: IWD-CO (Refinement)...")
    # Initialize IWD-CO with best positions from PPSO
    iwd_co = IWDCO(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        seed=seed + 1
    )
    iwd_co.initialise()
    
    # Transfer knowledge: set some water drops to best positions from PPSO
    if ppso.best:
        for i in range(min(5, len(iwd_co.population))):
            iwd_co.population[i].position = ppso.best.best_position[:]
            iwd_co.population[i].fitness = rastrigin(iwd_co.population[i].position)
            if iwd_co.population[i].fitness < iwd_co.best.fitness:
                iwd_co.best = iwd_co.population[i]
    
    phase2_iterations = iterations // 3
    for i in range(phase2_iterations):
        iwd_co.step()
    
    best_after_iwd = iwd_co.best.fitness if iwd_co.best else float('inf')
    print(f"  Best after IWD-CO: {best_after_iwd:.6f}")
    
    # Phase 3: Water Cycle for fine-tuning
    print("\nPhase 3: Water Cycle Algorithm (Fine-tuning)...")
    wca = WaterCycleAlgorithm(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        num_rivers=4,
        seed=seed + 2
    )
    wca.initialise()
    
    # Transfer knowledge: set sea to best from IWD-CO
    if iwd_co.best and wca.sea:
        wca.sea.position = iwd_co.best.position[:]
        wca.sea.fitness = rastrigin(wca.sea.position)
    
    phase3_iterations = iterations - phase1_iterations - phase2_iterations
    for i in range(phase3_iterations):
        wca.step()
    
    final_best = wca.sea.fitness if wca.sea else float('inf')
    print(f"  Final best (after all phases): {final_best:.6f}")
    
    print("\n" + "=" * 60)
    print("HYBRID APPROACH SUMMARY")
    print("=" * 60)
    print(f"Phase 1 (PPSO):     {best_after_ppso:.6f}")
    print(f"Phase 2 (IWD-CO):    {best_after_iwd:.6f}")
    print(f"Phase 3 (Water Cycle): {final_best:.6f}")
    print(f"Total Improvement:   {best_after_ppso - final_best:.6f}")
    
    return final_best


def main():
    """Main comparison example."""
    print("Algorithm Comparison: PPSO, IWD-CO, and Water Cycle Algorithm")
    print("=" * 60)
    
    # Configuration
    bounds = [(-5.12, 5.12)] * 5  # 5-dimensional Rastrigin
    population_size = 40
    iterations = 200
    seed = 42
    
    # Run comparison
    results = run_algorithm_comparison(
        bounds=bounds,
        population_size=population_size,
        iterations=iterations,
        seed=seed,
        verbose=True
    )
    
    # Print summary
    print_comparison_summary(results)
    
    # Plot convergence
    try:
        plot_convergence_comparison(results)
    except ImportError:
        print("\n⚠ matplotlib not available, skipping plot generation")
        print("  Install with: pip install matplotlib")
    
    # Demonstrate hybrid approach
    print("\n")
    hybrid_approach_example(
        bounds=bounds,
        population_size=population_size,
        iterations=iterations,
        seed=seed
    )


if __name__ == "__main__":
    main()

