"""
Adaptive Landscape module for the TRISOLARIS framework.

This module provides the AdaptiveLandscape class that models the fitness landscape
for navigating the solution space efficiently. It implements theoretical concepts
from evolutionary biology such as the Price equation and Fisher's Fundamental Theorem
to provide a rigorous mathematical foundation for evolutionary algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Any, Tuple, Dict, Union
from collections import defaultdict
import logging

# Import mathematical functions from evolutionary_math
from trisolaris.core.evolutionary_math import (
    calculate_price_equation,
    calculate_fisher_theorem,
    calculate_selection_gradient,
    calculate_fitness_landscape as characterize_fitness_landscape
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveLandscape:
    """
    Models the fitness landscape for code evolution.
    
    The AdaptiveLandscape class provides a mathematical foundation for evolutionary search
    by modeling the fitness landscape, computing gradients, and visualizing the search space.
    
    Theoretical Foundation:
    ----------------------
    This implementation is based on two fundamental concepts from evolutionary theory:
    
    1. Price Equation: A mathematical description of how trait values change from one
       generation to the next. It decomposes evolutionary change into:
       - Selection component: Covariance between fitness and trait value
       - Transmission component: Expected value of fitness times change in trait value
       
       Mathematically: ΔZ = Cov(w, z)/w̄ + E(w·Δz)/w̄
       
    2. Fisher's Fundamental Theorem: States that the rate of increase in fitness of a
       population is equal to the genetic variance in fitness at that time.
       
       Mathematically: ΔW = VA/W
       
    These theoretical foundations provide a rigorous basis for understanding how
    populations evolve in the fitness landscape and how selection pressure influences
    the direction and rate of evolution.
    """
    
    def __init__(self, fitness_function: Callable[[Any], float], dimensionality_reduction=None):
        """
        Initialize the Adaptive Landscape.
        
        Args:
            fitness_function: Function that evaluates individuals and returns fitness scores
            dimensionality_reduction: Optional function to reduce high-dimensional representations
                                    to 2D or 3D for visualization purposes
        """
        self.fitness_function = fitness_function
        self.dimensionality_reduction = dimensionality_reduction
        
        # History of the landscape exploration
        self.history = []
        
        # Cache for fitness evaluations to avoid redundant computation
        self.fitness_cache = {}
        
        # Default dimensionality reduction if none provided
        if self.dimensionality_reduction is None:
            # Simple PCA-like reduction (very basic)
            self.dimensionality_reduction = self._default_dimensionality_reduction
    
    def _default_dimensionality_reduction(self, population: List[Any], target_dims=2) -> np.ndarray:
        """
        Default dimensionality reduction for visualization.
        
        Args:
            population: List of individuals to project
            target_dims: Target dimensionality (2 or 3)
            
        Returns:
            Numpy array with projected coordinates
        """
        # This is an extremely simplified approach - in practice use PCA, t-SNE, or UMAP
        # We're just assigning random coordinates for demo purposes
        return np.random.rand(len(population), target_dims)
    
    def evaluate(self, individual: Any) -> float:
        """
        Evaluate an individual on the landscape.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            Fitness score
        """
        # Use cached value if available to avoid redundant computation
        individual_hash = self._hash_individual(individual)
        if individual_hash in self.fitness_cache:
            return self.fitness_cache[individual_hash]
        
        # Compute fitness
        fitness = self.fitness_function(individual)
        
        # Cache the result
        self.fitness_cache[individual_hash] = fitness
        
        return fitness
    
    def _hash_individual(self, individual: Any) -> str:
        """
        Create a hash representation of an individual for caching.
        
        Args:
            individual: The individual to hash
            
        Returns:
            String hash representation
        """
        # For code genomes, hash the source code
        if hasattr(individual, 'to_source'):
            return hash(individual.to_source())
        
        # Fall back to object id
        return id(individual)
    
    def get_gradient(self, individual: Any, step_size: float = 0.01, 
                    num_samples: int = 10) -> Any:
        """
        Estimate the gradient at a point in the landscape.
        
        This computes the direction of steepest ascent in the fitness landscape,
        which can guide the search toward higher fitness regions.
        
        Args:
            individual: The individual at which to compute the gradient
            step_size: Size of perturbations for finite difference approximation
            num_samples: Number of samples to use for gradient estimation
            
        Returns:
            Gradient vector or representation appropriate for the individual
        """
        # This is a simplified approach using finite differences
        # In practice, we'd need to specialize this for different genome types
        if hasattr(individual, 'get_numerical_representation'):
            # Use the numerical representation if available
            base_repr = individual.get_numerical_representation()
            base_fitness = self.evaluate(individual)
            
            # Initialize gradient
            gradient = np.zeros_like(base_repr)
            
            # Compute gradient using finite differences
            for i in range(len(base_repr)):
                # Perturb the ith component
                perturbed = base_repr.copy()
                perturbed[i] += step_size
                
                # Create a temporary individual with the perturbed representation
                temp_individual = individual.clone()
                temp_individual.set_from_numerical_representation(perturbed)
                
                # Evaluate the perturbed individual
                perturbed_fitness = self.evaluate(temp_individual)
                
                # Compute partial derivative
                gradient[i] = (perturbed_fitness - base_fitness) / step_size
            
            return gradient
        
        # If numerical representation is not available, use a more generic approach
        # by generating random perturbations and finding the best direction
        base_fitness = self.evaluate(individual)
        best_fitness_gain = 0
        best_mutation = None
        
        # Generate random perturbations
        for _ in range(num_samples):
            # Clone and mutate with a small probability
            mutated = individual.clone()
            mutated.mutate(rate=0.05)  # Small mutation rate
            
            # Evaluate the mutated individual
            mutated_fitness = self.evaluate(mutated)
            fitness_gain = mutated_fitness - base_fitness
            
            # Track the best mutation
            if fitness_gain > best_fitness_gain:
                best_fitness_gain = fitness_gain
                best_mutation = mutated
        
        # Return the best mutation as an approximation of the gradient direction
        return best_mutation if best_mutation else individual.clone()
    
    def hill_climb(self, starting_point: Any, steps: int = 10, 
                  step_size: float = 0.1) -> Tuple[Any, List[float]]:
        """
        Perform hill climbing on the landscape.
        
        Args:
            starting_point: Initial individual
            steps: Number of hill climbing steps
            step_size: Size of steps to take in gradient direction
            
        Returns:
            Tuple of (best individual found, list of fitness values)
        """
        current = starting_point.clone()
        fitness_history = [self.evaluate(current)]
        
        for _ in range(steps):
            # Compute gradient
            gradient = self.get_gradient(current)
            
            # If gradient is a genome (not a numerical vector)
            if not isinstance(gradient, np.ndarray):
                # Take the mutated genome as the new point
                current = gradient
            else:
                # Apply gradient if it's a numerical vector
                if hasattr(current, 'apply_gradient'):
                    current.apply_gradient(gradient, step_size)
            
            # Evaluate and track
            current_fitness = self.evaluate(current)
            fitness_history.append(current_fitness)
        
        return current, fitness_history
    
    def visualize(self, population: List[Any], title: str = "Fitness Landscape",
                 show_history: bool = False, ax=None) -> None:
        """
        Visualize the fitness landscape.
        
        Args:
            population: List of individuals to visualize
            title: Title for the plot
            show_history: Whether to show the history of landscape exploration
            ax: Optional matplotlib axis for the plot
        """
        # Evaluate all individuals
        fitness_values = [self.evaluate(ind) for ind in population]
        
        # Reduce dimensionality for visualization
        coords = self.dimensionality_reduction(population)
        
        # Create the visualization
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            if coords.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        # Plot the fitness landscape
        if coords.shape[1] == 3:
            sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                           c=fitness_values, cmap='viridis', s=50, alpha=0.8)
            ax.set_zlabel('Dimension 3')
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=fitness_values, 
                           cmap='viridis', s=50, alpha=0.8)
        
        # Add a colorbar
        plt.colorbar(sc, ax=ax, label='Fitness')
        
        # Show history if requested
        if show_history and self.history:
            history_coords = np.array([entry['coords'] for entry in self.history])
            if history_coords.shape[1] == 3:
                ax.plot(history_coords[:, 0], history_coords[:, 1], history_coords[:, 2], 
                       'r-', linewidth=2, alpha=0.7)
            else:
                ax.plot(history_coords[:, 0], history_coords[:, 1], 
                       'r-', linewidth=2, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def record_history(self, individual: Any, fitness: float = None) -> None:
        """
        Record a point in the landscape exploration history.
        
        Args:
            individual: The individual to record
            fitness: Optional precomputed fitness (computed if not provided)
        """
        if fitness is None:
            fitness = self.evaluate(individual)
        
        # Project to 2D or 3D for visualization
        if isinstance(individual, list):
            coords = self.dimensionality_reduction(individual)
        else:
            coords = self.dimensionality_reduction([individual])[0]
        
        # Record the entry
        self.history.append({
            'individual': individual,
            'fitness': fitness,
            'coords': coords
        })
    
    def identify_peaks(self, population: List[Any], 
                      neighborhood_size: float = 0.1) -> List[Any]:
        """
        Identify local optima (peaks) in the fitness landscape.
        
        Args:
            population: Population of individuals to analyze
            neighborhood_size: Size of neighborhood for local optimality check
            
        Returns:
            List of individuals representing local peaks
        """
        # Compute fitness for all individuals
        fitness_values = [self.evaluate(ind) for ind in population]
        
        # Project to a common space for distance calculation
        coords = self.dimensionality_reduction(population)
        
        # Find peaks (local optima)
        peaks = []
        for i, individual in enumerate(population):
            # Check if this individual has higher fitness than all neighbors
            is_peak = True
            for j, other in enumerate(population):
                if i == j:
                    continue
                
                # Calculate distance in the projected space
                distance = np.linalg.norm(coords[i] - coords[j])
                
                # If in neighborhood and has higher fitness, this is not a peak
                if distance < neighborhood_size and fitness_values[j] > fitness_values[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(individual)
        
        return peaks
    
    def analyze_landscape(self, population: List[Any], environment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the fitness landscape characteristics using advanced evolutionary mathematics.
        
        This method integrates theoretical concepts from evolutionary biology to provide
        a comprehensive analysis of the fitness landscape. It uses the mathematical functions
        from the evolutionary_math module to calculate key metrics that describe the landscape.
        
        Args:
            population: Population of individuals to analyze
            environment: Optional dictionary containing environmental parameters that affect fitness evaluation
            
        Returns:
            Dictionary with landscape metrics including:
            - Basic statistics (avg_fitness, max_fitness, min_fitness, etc.)
            - Landscape characteristics (ruggedness, num_peaks)
            - Theoretical metrics (selection_component, transmission_component, rate_of_increase)
            
        Note:
            The theoretical metrics provide insights into the evolutionary dynamics:
            - Selection component: Measures how selection is driving trait change
            - Transmission component: Measures how trait transmission affects evolution
            - Rate of increase: Predicted rate of fitness increase based on Fisher's theorem
        """
        # Compute fitness for all individuals
        fitness_values = [self.evaluate(ind) for ind in population]
        
        # Project to a common space for distance calculation
        coords = self.dimensionality_reduction(population)
        
        # Calculate basic statistics
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        fitness_variance = np.var(fitness_values)
        
        # Calculate ruggedness (approximate with autocorrelation)
        # This is a simplified approach - more sophisticated methods exist
        fitness_autocorr = np.corrcoef(fitness_values[:-1], fitness_values[1:])[0, 1] if len(fitness_values) > 1 else 0
        
        # Identify peaks
        peaks = self.identify_peaks(population)
        
        # Use the evolutionary_math functions for advanced analysis
        # For trait values, we'll use a simple metric like normalized fitness
        normalized_fitness = [f/max(fitness_values) if max(fitness_values) > 0 else 0 for f in fitness_values]
        
        # Calculate Price equation components
        try:
            total_change, selection_component, transmission_component = calculate_price_equation(
                population, fitness_values, normalized_fitness
            )
        except Exception as e:
            logger.warning(f"Error calculating Price equation: {e}")
            total_change, selection_component, transmission_component = 0.0, 0.0, 0.0
        
        # Calculate Fisher's theorem (using fitness variance as a proxy for additive genetic variance)
        try:
            rate_of_increase = calculate_fisher_theorem(
                population, fitness_values, fitness_variance
            )
        except Exception as e:
            logger.warning(f"Error calculating Fisher's theorem: {e}")
            rate_of_increase = 0.0
        
        # Use the characterize_fitness_landscape function for additional metrics
        if environment is None:
            environment = {}
        
        try:
            landscape_characteristics = characterize_fitness_landscape(population, environment)
        except Exception as e:
            logger.warning(f"Error characterizing fitness landscape: {e}")
            landscape_characteristics = {
                'ruggedness': 1 - fitness_autocorr,
                'num_peaks': len(peaks),
                'mean_gradient': 0.0,
                'max_fitness': max_fitness,
                'min_fitness': min_fitness,
                'fitness_range': max_fitness - min_fitness
            }
        
        # Combine all metrics
        result = {
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'fitness_variance': fitness_variance,
            'fitness_range': max_fitness - min_fitness,
            'ruggedness': landscape_characteristics.get('ruggedness', 1 - fitness_autocorr),
            'num_peaks': landscape_characteristics.get('num_peaks', len(peaks)),
            'peaks': peaks,
            # Theoretical metrics
            'price_equation': {
                'total_change': total_change,
                'selection_component': selection_component,
                'transmission_component': transmission_component
            },
            'fisher_theorem': {
                'rate_of_increase': rate_of_increase
            },
            'mean_gradient': landscape_characteristics.get('mean_gradient', 0.0)
        }
        
        return result
    
    def calculate_selection_pressure(self, population: List[Any]) -> Dict[str, Any]:
        """
        Calculate the selection pressure on the population using Fisher's Fundamental Theorem.
        
        Selection pressure is a key concept in evolutionary theory that quantifies how strongly
        natural selection is acting on a population. This method uses Fisher's Fundamental Theorem
        to calculate the rate of fitness increase, which is directly related to selection pressure.
        
        Theoretical Foundation:
        ----------------------
        Fisher's Fundamental Theorem states that the rate of increase in fitness of a population
        is equal to the genetic variance in fitness at that time:
        
        ΔW = VA/W
        
        Where:
        - ΔW is the rate of increase in fitness
        - VA is the additive genetic variance in fitness
        - W is the mean fitness
        
        A higher rate of increase indicates stronger selection pressure.
        
        Args:
            population: List of individuals to analyze
            
        Returns:
            Dictionary containing:
            - 'rate_of_increase': Predicted rate of fitness increase
            - 'selection_strength': Normalized measure of selection pressure (0-1)
            - 'mean_fitness': Average fitness of the population
            - 'fitness_variance': Variance in fitness values
        """
        # Compute fitness for all individuals
        fitness_values = [self.evaluate(ind) for ind in population]
        
        # Calculate mean fitness and variance
        mean_fitness = np.mean(fitness_values)
        fitness_variance = np.var(fitness_values)
        
        # Calculate rate of increase using Fisher's theorem
        try:
            rate_of_increase = calculate_fisher_theorem(
                population, fitness_values, fitness_variance
            )
        except Exception as e:
            logger.warning(f"Error calculating Fisher's theorem: {e}")
            rate_of_increase = 0.0
        
        # Normalize selection strength to a 0-1 scale
        # This is a simplified approach - in practice, this would be calibrated
        # based on the specific evolutionary system
        max_theoretical_rate = 1.0  # Theoretical maximum rate of increase
        selection_strength = min(rate_of_increase / max_theoretical_rate, 1.0) if max_theoretical_rate > 0 else 0.0
        
        return {
            'rate_of_increase': rate_of_increase,
            'selection_strength': selection_strength,
            'mean_fitness': mean_fitness,
            'fitness_variance': fitness_variance
        }
    
    def visualize_price_equation(self, population: List[Any], trait_name: str = "Fitness",
                               title: str = "Price Equation Components", ax=None) -> None:
        """
        Visualize the components of the Price equation for the population.
        
        The Price equation decomposes evolutionary change into selection and transmission
        components. This visualization helps understand how these components contribute
        to the overall evolutionary change.
        
        Args:
            population: List of individuals to analyze
            trait_name: Name of the trait being analyzed (for labeling)
            title: Title for the plot
            ax: Optional matplotlib axis for the plot
        """
        # Compute fitness for all individuals
        fitness_values = [self.evaluate(ind) for ind in population]
        
        # For trait values, we'll use a simple metric like normalized fitness
        normalized_fitness = [f/max(fitness_values) if max(fitness_values) > 0 else 0 for f in fitness_values]
        
        # Calculate Price equation components
        try:
            total_change, selection_component, transmission_component = calculate_price_equation(
                population, fitness_values, normalized_fitness
            )
        except Exception as e:
            logger.warning(f"Error calculating Price equation: {e}")
            total_change, selection_component, transmission_component = 0.0, 0.0, 0.0
        
        # Create the visualization
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a bar chart
        components = ['Total Change', 'Selection', 'Transmission']
        values = [total_change, selection_component, transmission_component]
        colors = ['blue', 'green', 'red']
        
        bars = ax.bar(components, values, color=colors, alpha=0.7)
        
        # Add labels and title
        ax.set_ylabel(f'Change in {trait_name}')
        ax.set_title(title)
        
        # Add text labels on the bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add explanatory text
        explanation = (
            "Price Equation Components:\n"
            "- Total Change: Overall change in trait value\n"
            "- Selection: Change due to fitness differences\n"
            "- Transmission: Change due to trait transmission"
        )
        ax.text(0.02, 0.02, explanation, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_fitness_landscape_3d(self, population: List[Any],
                                     title: str = "3D Fitness Landscape",
                                     show_gradients: bool = True) -> None:
        """
        Create a 3D visualization of the fitness landscape with selection gradients.
        
        This method creates a 3D surface plot representing the fitness landscape,
        with optional arrows showing the selection gradients (direction of steepest
        fitness increase).
        
        Args:
            population: List of individuals to visualize
            title: Title for the plot
            show_gradients: Whether to show selection gradient vectors
        """
        # Evaluate all individuals
        fitness_values = [self.evaluate(ind) for ind in population]
        
        # Reduce dimensionality to 3D for visualization
        coords = self.dimensionality_reduction(population, target_dims=3)
        
        # Create the 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a scatter plot of individuals
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=fitness_values, cmap='viridis', s=50, alpha=0.8)
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Fitness')
        
        # Show selection gradients if requested
        if show_gradients:
            # Calculate selection gradients
            try:
                gradients = calculate_selection_gradient(population, self.fitness_function)
                
                # Normalize gradients for visualization
                max_gradient = max(abs(g) for g in gradients) if gradients else 1.0
                normalized_gradients = [g / max_gradient if max_gradient > 0 else 0 for g in gradients]
                
                # Plot gradient arrows for a subset of points (to avoid clutter)
                stride = max(1, len(population) // 20)  # Show at most 20 arrows
                for i in range(0, len(population), stride):
                    if i < len(normalized_gradients):
                        # Create a simple arrow in the direction of increasing fitness
                        arrow_length = 0.1 * normalized_gradients[i]
                        ax.quiver(coords[i, 0], coords[i, 1], coords[i, 2],
                                arrow_length, arrow_length, arrow_length,
                                color='red', alpha=0.7)
            except Exception as e:
                logger.warning(f"Error calculating selection gradients: {e}")
        
        # Add labels and title
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(title)
        
        # Add a text box explaining the visualization
        explanation = (
            "Fitness Landscape Visualization:\n"
            "- Points represent individuals in the population\n"
            "- Color indicates fitness (brighter = higher fitness)\n"
            "- Red arrows show selection gradients (direction of fitness increase)"
        )
        ax.text2D(0.02, 0.02, explanation, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()