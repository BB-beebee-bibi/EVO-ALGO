"""
Adaptive Landscape module for the TRISOLARIS framework.

This module provides the AdaptiveLandscape class that models the fitness landscape
for navigating the solution space efficiently.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Any, Tuple, Dict
from collections import defaultdict

class AdaptiveLandscape:
    """
    Models the fitness landscape for code evolution.
    
    The AdaptiveLandscape class provides a mathematical foundation for evolutionary search
    by modeling the fitness landscape, computing gradients, and visualizing the search space.
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
    
    def analyze_landscape(self, population: List[Any]) -> Dict[str, Any]:
        """
        Analyze the fitness landscape characteristics.
        
        Args:
            population: Population of individuals to analyze
            
        Returns:
            Dictionary with landscape metrics
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
        fitness_autocorr = np.corrcoef(fitness_values[:-1], fitness_values[1:])[0, 1]
        
        # Identify peaks
        peaks = self.identify_peaks(population)
        
        # Return metrics
        return {
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'fitness_variance': fitness_variance,
            'fitness_range': max_fitness - min_fitness,
            'ruggedness': 1 - fitness_autocorr,  # Higher value = more rugged
            'num_peaks': len(peaks),
            'peaks': peaks
        } 