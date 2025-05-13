"""
Population diversity metrics and visualization module for the TRISOLARIS framework.

This module provides functions for calculating and visualizing population diversity
metrics, which are important indicators of the health and progress of evolutionary processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_genotypic_diversity(population: List[Any]) -> float:
    """
    Calculate genotypic diversity of a population.
    
    This function calculates diversity based on the genetic representation of individuals.
    Higher values indicate more diverse populations.
    
    Args:
        population: List of individuals to analyze
        
    Returns:
        Diversity score between 0 and 1
    """
    if not population or len(population) < 2:
        return 0.0
    
    try:
        # Get string representations of genomes
        genomes = [str(ind) for ind in population]
        
        # Calculate pairwise differences
        total_diff = 0
        count = 0
        
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                # Use Levenshtein distance if available, otherwise use simple comparison
                try:
                    import Levenshtein
                    distance = Levenshtein.distance(genomes[i], genomes[j])
                    max_len = max(len(genomes[i]), len(genomes[j]))
                    if max_len > 0:
                        total_diff += distance / max_len
                except ImportError:
                    # Simple comparison: proportion of characters that differ
                    min_len = min(len(genomes[i]), len(genomes[j]))
                    if min_len > 0:
                        diff_count = sum(1 for a, b in zip(genomes[i][:min_len], genomes[j][:min_len]) if a != b)
                        total_diff += diff_count / min_len
                
                count += 1
        
        # Normalize diversity score
        if count > 0:
            return total_diff / count
        return 0.0
    
    except Exception as e:
        logger.warning(f"Error calculating genotypic diversity: {e}")
        return 0.0

def calculate_phenotypic_diversity(population: List[Any], fitness_function) -> float:
    """
    Calculate phenotypic diversity of a population.
    
    This function calculates diversity based on the phenotypic expression (behavior)
    of individuals. Higher values indicate more diverse behaviors.
    
    Args:
        population: List of individuals to analyze
        fitness_function: Function to evaluate individuals
        
    Returns:
        Diversity score between 0 and 1
    """
    if not population or len(population) < 2:
        return 0.0
    
    try:
        # Evaluate all individuals
        fitness_values = [fitness_function(ind) for ind in population]
        
        # Calculate coefficient of variation (normalized standard deviation)
        std_dev = np.std(fitness_values)
        mean = np.mean(fitness_values)
        
        if mean != 0:
            cv = std_dev / abs(mean)
            # Normalize to [0, 1] using a sigmoid-like function
            diversity = 2 * (1 / (1 + np.exp(-cv))) - 1
            return max(0.0, min(1.0, diversity))
        return 0.0
    
    except Exception as e:
        logger.warning(f"Error calculating phenotypic diversity: {e}")
        return 0.0

def calculate_structural_diversity(population: List[Any]) -> float:
    """
    Calculate structural diversity of a population.
    
    This function calculates diversity based on the structure of individuals,
    such as code structure for code evolution. Higher values indicate more diverse structures.
    
    Args:
        population: List of individuals to analyze
        
    Returns:
        Diversity score between 0 and 1
    """
    if not population or len(population) < 2:
        return 0.0
    
    try:
        # Try to extract structural features (e.g., AST nodes for code)
        import ast
        
        # Count AST node types for each individual
        node_type_counts = []
        for ind in population:
            try:
                tree = ast.parse(str(ind))
                node_types = {}
                for node in ast.walk(tree):
                    node_type = type(node).__name__
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                node_type_counts.append(node_types)
            except:
                # If parsing fails, use an empty dict
                node_type_counts.append({})
        
        # Calculate pairwise Jaccard distances between node type distributions
        total_diff = 0
        count = 0
        
        for i in range(len(node_type_counts)):
            for j in range(i + 1, len(node_type_counts)):
                types_i = set(node_type_counts[i].keys())
                types_j = set(node_type_counts[j].keys())
                
                # Jaccard distance: 1 - (intersection size / union size)
                intersection = types_i.intersection(types_j)
                union = types_i.union(types_j)
                
                if union:
                    jaccard_dist = 1 - (len(intersection) / len(union))
                    total_diff += jaccard_dist
                
                count += 1
        
        # Normalize diversity score
        if count > 0:
            return total_diff / count
        return 0.0
    
    except Exception as e:
        logger.warning(f"Error calculating structural diversity: {e}")
        return 0.0

def track_diversity_metrics(
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
    # Initialize diversity_metrics list if it doesn't exist
    if 'diversity_metrics' not in metrics:
        metrics['diversity_metrics'] = []
    
    # Calculate diversity metrics
    genotypic_diversity = calculate_genotypic_diversity(population)
    phenotypic_diversity = calculate_phenotypic_diversity(population, fitness_function)
    structural_diversity = calculate_structural_diversity(population)
    
    # Add metrics for this generation
    metrics['diversity_metrics'].append({
        'generation': generation,
        'genotypic_diversity': genotypic_diversity,
        'phenotypic_diversity': phenotypic_diversity,
        'structural_diversity': structural_diversity
    })
    
    return metrics

def visualize_diversity_metrics(
    metrics: Dict[str, Any],
    output_path: str,
    interactive: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """
    Visualize diversity metrics over generations.
    
    Args:
        metrics: Dictionary containing diversity metrics
        output_path: Path to save the visualization
        interactive: Whether to create an interactive visualization
        figsize: Figure size as (width, height)
        
    Returns:
        Path to the saved visualization file
    """
    # Check if diversity metrics exist
    if 'diversity_metrics' not in metrics or not metrics['diversity_metrics']:
        # Fall back to using fitness history standard deviation as a simple diversity metric
        if 'fitness_history' in metrics and metrics['fitness_history']:
            generations = [m['generation'] for m in metrics['fitness_history']]
            diversity = [m.get('std_dev', 0) for m in metrics['fitness_history']]
            
            if interactive:
                try:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=generations, 
                        y=diversity,
                        mode='lines+markers',
                        name='Fitness Diversity (StdDev)',
                        line=dict(color='purple', width=2)
                    ))
                    
                    fig.update_layout(
                        title='Population Diversity Over Generations',
                        xaxis_title='Generation',
                        yaxis_title='Diversity (StdDev)',
                        template='plotly_white'
                    )
                    
                    html_path = str(Path(output_path).with_suffix('.html'))
                    fig.write_html(html_path, include_plotlyjs='cdn')
                    return html_path
                except Exception as e:
                    logger.warning(f"Error creating interactive plot: {e}")
                    interactive = False
            
            # Static plot
            plt.figure(figsize=figsize)
            plt.plot(generations, diversity, 'purple', marker='o', label='Fitness Diversity (StdDev)')
            plt.title('Population Diversity Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Diversity (StdDev)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        
        logger.warning("No diversity metrics or fitness history found")
        return None
    
    # Extract data
    generations = [m['generation'] for m in metrics['diversity_metrics']]
    genotypic = [m.get('genotypic_diversity', 0) for m in metrics['diversity_metrics']]
    phenotypic = [m.get('phenotypic_diversity', 0) for m in metrics['diversity_metrics']]
    structural = [m.get('structural_diversity', 0) for m in metrics['diversity_metrics']]
    
    if interactive:
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=generations, 
                y=genotypic,
                mode='lines+markers',
                name='Genotypic Diversity',
                line=dict(color='purple', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=generations, 
                y=phenotypic,
                mode='lines+markers',
                name='Phenotypic Diversity',
                line=dict(color='orange', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=generations, 
                y=structural,
                mode='lines+markers',
                name='Structural Diversity',
                line=dict(color='brown', width=2)
            ))
            
            fig.update_layout(
                title='Population Diversity Metrics Over Generations',
                xaxis_title='Generation',
                yaxis_title='Diversity',
                yaxis=dict(range=[0, 1]),
                template='plotly_white'
            )
            
            html_path = str(Path(output_path).with_suffix('.html'))
            fig.write_html(html_path, include_plotlyjs='cdn')
            return html_path
        except Exception as e:
            logger.warning(f"Error creating interactive plot: {e}")
            interactive = False
    
    # Static plot
    plt.figure(figsize=figsize)
    plt.plot(generations, genotypic, 'purple', marker='o', label='Genotypic Diversity')
    plt.plot(generations, phenotypic, 'orange', marker='s', label='Phenotypic Diversity')
    plt.plot(generations, structural, 'brown', marker='^', label='Structural Diversity')
    plt.title('Population Diversity Metrics Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_diversity_heatmap(
    population: List[Any],
    output_path: str,
    interactive: bool = False,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """
    Create a heatmap visualization of population diversity.
    
    Args:
        population: List of individuals to analyze
        output_path: Path to save the visualization
        interactive: Whether to create an interactive visualization
        figsize: Figure size as (width, height)
        
    Returns:
        Path to the saved visualization file
    """
    if not population or len(population) < 2:
        logger.warning("Population too small for diversity heatmap")
        return None
    
    try:
        # Calculate pairwise distances
        n = len(population)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # Use Levenshtein distance if available, otherwise use simple comparison
                    try:
                        import Levenshtein
                        distance = Levenshtein.distance(str(population[i]), str(population[j]))
                        max_len = max(len(str(population[i])), len(str(population[j])))
                        if max_len > 0:
                            distance_matrix[i, j] = distance / max_len
                    except ImportError:
                        # Simple comparison: proportion of characters that differ
                        s1 = str(population[i])
                        s2 = str(population[j])
                        min_len = min(len(s1), len(s2))
                        if min_len > 0:
                            diff_count = sum(1 for a, b in zip(s1[:min_len], s2[:min_len]) if a != b)
                            distance_matrix[i, j] = diff_count / min_len
        
        if interactive:
            try:
                import plotly.figure_factory as ff
                
                # Create heatmap
                fig = ff.create_annotated_heatmap(
                    z=distance_matrix,
                    x=[f'Ind {i+1}' for i in range(n)],
                    y=[f'Ind {i+1}' for i in range(n)],
                    colorscale='Viridis',
                    showscale=True
                )
                
                fig.update_layout(
                    title='Population Diversity Heatmap',
                    xaxis_title='Individual',
                    yaxis_title='Individual',
                    width=figsize[0] * 100,
                    height=figsize[1] * 100
                )
                
                html_path = str(Path(output_path).with_suffix('.html'))
                fig.write_html(html_path, include_plotlyjs='cdn')
                return html_path
            except Exception as e:
                logger.warning(f"Error creating interactive heatmap: {e}")
                interactive = False
        
        # Static plot
        plt.figure(figsize=figsize)
        plt.imshow(distance_matrix, cmap='viridis')
        plt.colorbar(label='Normalized Distance')
        plt.title('Population Diversity Heatmap')
        plt.xlabel('Individual')
        plt.ylabel('Individual')
        plt.xticks(range(n), [f'Ind {i+1}' for i in range(n)], rotation=90)
        plt.yticks(range(n), [f'Ind {i+1}' for i in range(n)])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.warning(f"Error creating diversity heatmap: {e}")
        return None