"""
Visualization module for the TRISOLARIS framework.

This module provides visualization utilities for evolutionary processes,
including fitness progression, resource usage, population diversity,
ethics evaluation, syntax error rates, and other metrics.

The module supports both static visualizations using matplotlib and
interactive visualizations using Plotly.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import submodules
try:
    from . import interactive
    from . import diversity
    from . import syntax_errors
    from . import dashboard
    HAS_INTERACTIVE = True
except ImportError:
    logger.warning("Interactive visualization dependencies not found. Install plotly for interactive visualizations.")
    HAS_INTERACTIVE = False

def visualize_evolution_progress(
    generations: List[int],
    fitness_values: Dict[str, List[float]],
    output_path: str,
    title: str = "Evolution Progress",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize the evolution progress over generations.
    
    Args:
        generations: List of generation numbers
        fitness_values: Dictionary mapping metric names to lists of values
        output_path: Path to save the visualization
        title: Title for the plot
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    for label, values in fitness_values.items():
        plt.plot(generations, values, label=label)
    
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    logger.info(f"Saved evolution progress visualization to {output_path}")

def visualize_resource_usage(
    generations: List[int],
    cpu_usage: List[float],
    memory_usage: List[float],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize resource usage during evolution.
    
    Args:
        generations: List of generation numbers
        cpu_usage: List of CPU usage percentages
        memory_usage: List of memory usage percentages
        output_path: Path to save the visualization
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    plt.plot(generations, cpu_usage, label="CPU Usage (%)")
    plt.plot(generations, memory_usage, label="Memory Usage (%)")
    
    plt.title("Resource Usage During Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Usage (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    logger.info(f"Saved resource usage visualization to {output_path}")

def visualize_ethics_evaluation(
    categories: List[str],
    scores: List[float],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize ethics evaluation results.
    
    Args:
        categories: List of ethical categories
        scores: List of scores for each category
        output_path: Path to save the visualization
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    plt.bar(categories, scores)
    plt.title("Ethics Evaluation by Category")
    plt.xlabel("Category")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    logger.info(f"Saved ethics evaluation visualization to {output_path}")

def visualize_selection_pressure(
    generations: List[int],
    selection_components: List[float],
    transmission_components: List[float],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize selection pressure metrics from Price equation.
    
    Args:
        generations: List of generation numbers
        selection_components: List of selection components from Price equation
        transmission_components: List of transmission components from Price equation
        output_path: Path to save the visualization
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    plt.plot(generations, selection_components, label="Selection Component")
    plt.plot(generations, transmission_components, label="Transmission Component")
    plt.plot(generations, [s + t for s, t in zip(selection_components, transmission_components)], 
             label="Total Change", linestyle="--")
    
    plt.title("Selection Pressure (Price Equation Components)")
    plt.xlabel("Generation")
    plt.ylabel("Component Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    logger.info(f"Saved selection pressure visualization to {output_path}")

def create_dashboard(
    metrics: Dict[str, Any],
    output_path: str,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create a comprehensive dashboard of evolution metrics.
    
    Args:
        metrics: Dictionary containing all evolution metrics
        output_path: Path to save the visualization
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Extract data
    generations = [m['generation'] for m in metrics.get('fitness_history', [])]
    
    if not generations:
        logger.warning("No generation data available for dashboard")
        return
    
    # Plot fitness progression
    plt.subplot(2, 2, 1)
    best_fitness = [m['best_fitness'] for m in metrics.get('fitness_history', [])]
    avg_fitness = [m['avg_fitness'] for m in metrics.get('fitness_history', [])]
    min_fitness = [m['min_fitness'] for m in metrics.get('fitness_history', [])]
    
    plt.plot(generations, best_fitness, 'b-', label='Best Fitness')
    plt.plot(generations, avg_fitness, 'g-', label='Average Fitness')
    plt.plot(generations, min_fitness, 'r-', label='Min Fitness')
    plt.title('Fitness Progression')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    # Plot Price equation components
    if metrics.get('price_equation'):
        plt.subplot(2, 2, 2)
        total_change = [m['total_change'] for m in metrics.get('price_equation', [])]
        selection = [m['selection_component'] for m in metrics.get('price_equation', [])]
        transmission = [m['transmission_component'] for m in metrics.get('price_equation', [])]
        
        plt.plot(generations, total_change, 'b-', label='Total Change')
        plt.plot(generations, selection, 'g-', label='Selection')
        plt.plot(generations, transmission, 'r-', label='Transmission')
        plt.title('Price Equation Components')
        plt.xlabel('Generation')
        plt.ylabel('Component Value')
        plt.legend()
        plt.grid(True)
    
    # Plot resource usage
    if metrics.get('resource_usage'):
        plt.subplot(2, 2, 3)
        cpu_usage = [m.get('cpu_percent', 0) for m in metrics.get('resource_usage', [])]
        memory_usage = [m.get('memory_percent', 0) for m in metrics.get('resource_usage', [])]
        
        plt.plot(generations, cpu_usage, 'b-', label='CPU Usage (%)')
        plt.plot(generations, memory_usage, 'g-', label='Memory Usage (%)')
        plt.title('Resource Usage')
        plt.xlabel('Generation')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True)
    
    # Plot fitness landscape characteristics
    if metrics.get('fitness_landscape'):
        plt.subplot(2, 2, 4)
        ruggedness = [m['ruggedness'] for m in metrics.get('fitness_landscape', [])]
        num_peaks = [m['num_peaks'] for m in metrics.get('fitness_landscape', [])]
        
        plt.plot(generations, ruggedness, 'b-', label='Landscape Ruggedness')
        plt.plot(generations, num_peaks, 'g-', label='Number of Peaks')
        plt.title('Fitness Landscape Characteristics')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved evolution dashboard to {output_path}")
    
    return output_path

def create_visualization_dashboard(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    interactive: bool = True
) -> str:
    """
    Create a comprehensive visualization dashboard.
    
    This function creates a dashboard with multiple visualizations of
    evolutionary metrics, with support for both static and interactive
    visualizations.
    
    Args:
        metrics: Dictionary containing evolution metrics
        output_dir: Directory to save visualizations
        interactive: Whether to use interactive visualizations (requires plotly)
        
    Returns:
        Path to the saved dashboard file
    """
    if interactive and HAS_INTERACTIVE:
        # Use the EvolutionDashboard class for interactive visualizations
        dash = dashboard.EvolutionDashboard(metrics, output_dir, interactive=True)
        return dash.create_dashboard()
    else:
        # Fall back to static dashboard
        output_path = str(Path(output_dir) / "dashboard.png")
        return create_dashboard(metrics, output_path)

def track_diversity(
    population: List[Any],
    fitness_function,
    generation: int,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Track diversity metrics for a population.
    
    Args:
        population: List of individuals to analyze
        fitness_function: Function to evaluate individuals
        generation: Current generation number
        metrics: Dictionary to update with diversity metrics
        
    Returns:
        Updated metrics dictionary
    """
    if HAS_INTERACTIVE:
        return diversity.track_diversity_metrics(population, fitness_function, generation, metrics)
    return metrics

def track_syntax_errors(
    generation: int,
    population_size: int,
    error_count: int,
    repair_success_count: int,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Track syntax error metrics for a generation.
    
    Args:
        generation: Current generation number
        population_size: Size of the population
        error_count: Number of syntax errors encountered
        repair_success_count: Number of successful repairs
        metrics: Dictionary to update with syntax error metrics
        
    Returns:
        Updated metrics dictionary
    """
    if HAS_INTERACTIVE:
        return syntax_errors.track_syntax_errors(
            generation, population_size, error_count, repair_success_count, metrics
        )
    return metrics

def export_visualization_data(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    formats: List[str] = None
) -> Dict[str, str]:
    """
    Export visualization data to various formats.
    
    Args:
        metrics: Dictionary containing evolution metrics
        output_dir: Directory to save exported data
        formats: List of formats to export (default: ['csv', 'json'])
        
    Returns:
        Dictionary mapping data type to file path
    """
    if HAS_INTERACTIVE:
        dash = dashboard.EvolutionDashboard(metrics, output_dir, interactive=False)
        return dash.export_data(formats)
    
    # Simple JSON export if dashboard module is not available
    import json
    output_path = Path(output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    return {"metrics_json": str(output_path)}