"""
Evolutionary Mathematics module for the TRISOLARIS framework.

This module implements core mathematical functions from evolutionary theory
that provide the theoretical foundation for the evolutionary algorithms.
It bridges theoretical evolutionary biology with computational implementations.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_price_equation(population: List[Any], 
                            fitness_values: List[float], 
                            trait_values: List[float]) -> Tuple[float, float, float]:
    """
    Calculate the Price equation components for a given population.
    
    The Price equation is a fundamental mathematical theorem that describes how the average 
    value of a trait changes from one generation to the next. It decomposes evolutionary 
    change into two components:
    
    1. Selection component: Covariance between fitness and trait value
    2. Transmission component: Expected value of fitness times change in trait value
    
    Mathematically expressed as: ΔZ = Cov(w, z)/w̄ + E(w·Δz)/w̄
    
    Where:
    - ΔZ is the change in the average value of trait z
    - w is fitness
    - z is the trait value
    - w̄ is the average fitness
    - Cov(w, z) is the covariance between fitness and trait value
    - E(w·Δz) is the expected value of the product of fitness and the change in trait value
    
    This implementation assumes that the trait values are already extracted from the population
    and that fitness values have been calculated. For code genomes, traits might be code metrics
    like complexity, length, or specific structural features.
    
    Args:
        population: List of individuals (genomes)
        fitness_values: List of fitness values corresponding to each individual
        trait_values: List of trait values to analyze
        
    Returns:
        Tuple containing:
        - total_change: The total change in the average trait value (ΔZ)
        - selection_component: The component due to selection (Cov(w, z)/w̄)
        - transmission_component: The component due to transmission bias (E(w·Δz)/w̄)
        
    Example:
        >>> population = [genome1, genome2, genome3]
        >>> fitness_values = [0.8, 0.9, 0.7]
        >>> trait_values = [10, 15, 8]  # e.g., code complexity
        >>> total, selection, transmission = calculate_price_equation(population, fitness_values, trait_values)
        >>> print(f"Total change: {total}, Selection: {selection}, Transmission: {transmission}")
    """
    # Validate inputs
    if len(population) != len(fitness_values) or len(population) != len(trait_values):
        raise ValueError("Population, fitness_values, and trait_values must have the same length")
    
    if not population:
        return 0.0, 0.0, 0.0
    
    # Calculate mean fitness
    mean_fitness = np.mean(fitness_values)
    if mean_fitness == 0:
        logger.warning("Mean fitness is zero, cannot calculate Price equation")
        return 0.0, 0.0, 0.0
    
    # Calculate mean trait value
    mean_trait = np.mean(trait_values)
    
    # Calculate covariance between fitness and trait value
    # This represents the selection component
    covariance = np.cov(fitness_values, trait_values)[0, 1]
    selection_component = covariance / mean_fitness
    
    # For the transmission component, we need to estimate the change in trait value
    # In a real system, this would be measured from parent to offspring
    # Here we'll use a simplified approach assuming some transmission bias
    
    # Simulate transmission bias (in a real system, this would be measured)
    # For this example, we'll assume transmission bias is proportional to fitness
    # Higher fitness individuals have more accurate transmission (less bias)
    transmission_bias = [0.01 * (1 - f/max(fitness_values)) * t for f, t in zip(fitness_values, trait_values)]
    
    # Calculate the expected value of fitness times transmission bias
    transmission_component = np.mean([f * b for f, b in zip(fitness_values, transmission_bias)]) / mean_fitness
    
    # Total change is the sum of both components
    total_change = selection_component + transmission_component
    
    return total_change, selection_component, transmission_component


def calculate_fisher_theorem(population: List[Any], 
                            fitness_values: List[float], 
                            additive_genetic_variance: float) -> float:
    """
    Calculate the rate of increase in fitness according to Fisher's Fundamental Theorem.
    
    Fisher's Fundamental Theorem of Natural Selection states that the rate of increase in 
    fitness of a population at any time is equal to the genetic variance in fitness at that time.
    
    Mathematically expressed as: ΔW = VA/W
    
    Where:
    - ΔW is the rate of increase in fitness
    - VA is the additive genetic variance in fitness
    - W is the mean fitness
    
    This theorem provides a quantitative prediction of how quickly a population will adapt
    through natural selection. It's a cornerstone of population genetics and evolutionary theory.
    
    Args:
        population: List of individuals (genomes)
        fitness_values: List of fitness values corresponding to each individual
        additive_genetic_variance: The additive genetic variance in fitness
            (This is typically estimated from parent-offspring regression or other methods)
        
    Returns:
        The predicted rate of increase in fitness (ΔW)
        
    Example:
        >>> population = [genome1, genome2, genome3]
        >>> fitness_values = [0.8, 0.9, 0.7]
        >>> additive_genetic_variance = 0.05  # Estimated from population data
        >>> rate_of_increase = calculate_fisher_theorem(population, fitness_values, additive_genetic_variance)
        >>> print(f"Predicted rate of fitness increase: {rate_of_increase}")
    """
    # Validate inputs
    if len(population) != len(fitness_values):
        raise ValueError("Population and fitness_values must have the same length")
    
    if not population:
        return 0.0
    
    # Calculate mean fitness
    mean_fitness = np.mean(fitness_values)
    if mean_fitness == 0:
        logger.warning("Mean fitness is zero, cannot calculate Fisher's theorem")
        return 0.0
    
    # Apply Fisher's Fundamental Theorem
    rate_of_increase = additive_genetic_variance / mean_fitness
    
    return rate_of_increase


def calculate_selection_gradient(population: List[Any], 
                                fitness_landscape: Callable[[Any], float]) -> List[float]:
    """
    Calculate the selection gradient for a population in a given fitness landscape.
    
    The selection gradient represents the direction and magnitude of selection acting on 
    different traits. It is a vector of partial derivatives of fitness with respect to 
    each trait, indicating how fitness would change with small changes in trait values.
    
    In evolutionary computation, this can guide the search toward promising regions of 
    the solution space by indicating which traits should be modified to increase fitness.
    
    Args:
        population: List of individuals (genomes)
        fitness_landscape: A function that maps an individual to its fitness in the landscape
        
    Returns:
        A list of selection gradients for each individual in the population
        
    Example:
        >>> def fitness_function(genome):
        ...     # Example fitness function based on code complexity and correctness
        ...     return 0.7 * genome.correctness - 0.3 * genome.complexity
        >>> 
        >>> population = [genome1, genome2, genome3]
        >>> gradients = calculate_selection_gradient(population, fitness_function)
        >>> for i, gradient in enumerate(gradients):
        ...     print(f"Individual {i}: Selection gradient = {gradient}")
    """
    if not population:
        return []
    
    gradients = []
    
    for individual in population:
        # Calculate baseline fitness
        baseline_fitness = fitness_landscape(individual)
        
        # For code genomes, we can't easily compute analytical gradients
        # Instead, we'll use a numerical approximation by applying small mutations
        # and measuring the fitness change
        
        # Clone the individual to avoid modifying the original
        if hasattr(individual, 'clone'):
            clone = individual.clone()
            
            # Apply a small mutation
            if hasattr(clone, 'mutate'):
                clone.mutate(rate=0.01)  # Small mutation rate
                
                # Calculate fitness after mutation
                mutated_fitness = fitness_landscape(clone)
                
                # Calculate approximate gradient
                gradient = mutated_fitness - baseline_fitness
                gradients.append(gradient)
            else:
                logger.warning(f"Individual {individual} does not have a mutate method")
                gradients.append(0.0)
        else:
            logger.warning(f"Individual {individual} does not have a clone method")
            gradients.append(0.0)
    
    return gradients


def calculate_fitness_landscape(population: List[Any], 
                               environment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate and characterize the fitness landscape for a population in a given environment.
    
    The fitness landscape is a conceptual tool that maps genotypes to fitness values, creating
    a multidimensional surface where peaks represent high fitness and valleys represent low fitness.
    Understanding the structure of this landscape provides insights into evolutionary dynamics.
    
    This function analyzes the landscape's key properties such as ruggedness, number of peaks,
    and gradient information, which can inform evolutionary algorithm parameters and strategies.
    
    Args:
        population: List of individuals (genomes)
        environment: Dictionary containing environmental parameters that affect fitness evaluation
        
    Returns:
        Dictionary containing fitness landscape characteristics:
        - 'ruggedness': Measure of landscape complexity (higher = more rugged)
        - 'num_peaks': Estimated number of local optima
        - 'mean_gradient': Average selection gradient magnitude
        - 'max_fitness': Maximum fitness in the current population
        - 'min_fitness': Minimum fitness in the current population
        - 'fitness_range': Range of fitness values (max - min)
        
    Example:
        >>> population = [genome1, genome2, genome3]
        >>> environment = {'mutation_rate': 0.1, 'selection_pressure': 0.7}
        >>> landscape_info = calculate_fitness_landscape(population, environment)
        >>> print(f"Landscape ruggedness: {landscape_info['ruggedness']}")
        >>> print(f"Estimated number of peaks: {landscape_info['num_peaks']}")
    """
    if not population:
        return {
            'ruggedness': 0.0,
            'num_peaks': 0,
            'mean_gradient': 0.0,
            'max_fitness': 0.0,
            'min_fitness': 0.0,
            'fitness_range': 0.0
        }
    
    # Define a fitness function based on the environment
    def fitness_function(individual):
        # This is a placeholder - in a real system, this would use the environment
        # parameters to evaluate the individual
        if hasattr(individual, 'to_source'):
            code = individual.to_source()
            # Simple metrics: code length and estimated complexity
            length = len(code)
            complexity = code.count('if') + code.count('for') + code.count('while')
            
            # Example fitness calculation using environment parameters
            mutation_rate = environment.get('mutation_rate', 0.1)
            selection_pressure = environment.get('selection_pressure', 0.5)
            
            # Higher selection pressure favors shorter, less complex code
            length_penalty = length * 0.01 * selection_pressure
            complexity_bonus = complexity * 0.05 * (1 - selection_pressure)
            
            return 1.0 + complexity_bonus - length_penalty
        else:
            return 0.5  # Default fitness for non-code individuals
    
    # Calculate fitness for all individuals
    fitness_values = [fitness_function(ind) for ind in population]
    
    # Calculate basic statistics
    max_fitness = max(fitness_values)
    min_fitness = min(fitness_values)
    fitness_range = max_fitness - min_fitness
    
    # Calculate selection gradients
    gradients = calculate_selection_gradient(population, fitness_function)
    mean_gradient = np.mean([abs(g) for g in gradients])
    
    # Estimate landscape ruggedness
    # Higher variance in gradients suggests a more rugged landscape
    gradient_variance = np.var(gradients) if len(gradients) > 1 else 0
    ruggedness = np.sqrt(gradient_variance)
    
    # Estimate number of peaks (local optima)
    # A simple heuristic: count individuals with higher fitness than their neighbors
    num_peaks = 0
    for i, ind in enumerate(population):
        is_peak = True
        ind_fitness = fitness_values[i]
        
        # Check if this individual has higher fitness than similar individuals
        for j, other in enumerate(population):
            if i == j:
                continue
            
            # Skip distant individuals (assuming some distance metric)
            # For simplicity, we'll just check a few neighbors
            if abs(i - j) > 3:
                continue
            
            if fitness_values[j] > ind_fitness:
                is_peak = False
                break
        
        if is_peak:
            num_peaks += 1
    
    return {
        'ruggedness': ruggedness,
        'num_peaks': num_peaks,
        'mean_gradient': mean_gradient,
        'max_fitness': max_fitness,
        'min_fitness': min_fitness,
        'fitness_range': fitness_range
    }