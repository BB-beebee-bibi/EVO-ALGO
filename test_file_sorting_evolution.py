"""
Test script demonstrating the scaffolded evolution approach for file sorting tasks.
"""

import ast
import astor
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from trisolaris.operators import ExonMutator, CodeValidator, ExonEvolutionEngine
from trisolaris.utils.ast_utils import get_function_nodes

# Initial population with functional building blocks
INITIAL_POPULATION = [
    # Basic file listing
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    return sorted(files)  # Simple alphabetical sort
    """),
    
    # Extension-based sorting
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    return sorted(files, key=lambda f: os.path.splitext(f)[1])
    """),
    
    # Date-based sorting
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    """),
    
    # Size-based sorting
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    return sorted(files, key=lambda f: os.path.getsize(os.path.join(directory, f)))
    """),
    
    # Combined sorting (extension then date)
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    return sorted(files, key=lambda f: (os.path.splitext(f)[1], 
                                      os.path.getmtime(os.path.join(directory, f))))
    """)
]

def evaluate_file_sorting_function(ast_node: ast.AST, task_type: str = 'extension') -> float:
    """
    Evaluate a file sorting function's fitness with graduated scoring.
    
    Args:
        ast_node: AST of the file sorting function
        task_type: Type of sorting task ('extension', 'date', 'size', 'combined')
        
    Returns:
        Fitness score based on correctness and efficiency
    """
    try:
        # Convert AST to source code
        source = astor.to_source(ast_node)
        
        # Create a namespace for execution
        namespace = {}
        exec(source, namespace)
        
        # Get the sorting function
        sort_func = namespace['sort_files']
        
        # Create test directory with sample files
        test_dir = "test_files"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            # Create files with different extensions, sizes, and dates
            extensions = ['.txt', '.pdf', '.py', '.json']
            for i in range(10):
                ext = random.choice(extensions)
                with open(os.path.join(test_dir, f"file{i}{ext}"), "w") as f:
                    f.write("x" * (i * 100))  # Different sizes
                # Set modification time to be different for each file
                os.utime(os.path.join(test_dir, f"file{i}{ext}"), (0, i * 1000))
        
        # Get list of files
        files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
        
        # Calculate expected order based on task type
        if task_type == 'extension':
            expected = sorted(files, key=lambda f: os.path.splitext(f)[1])
        elif task_type == 'date':
            expected = sorted(files, key=os.path.getmtime)
        elif task_type == 'size':
            expected = sorted(files, key=os.path.getsize)
        else:  # combined
            expected = sorted(files, key=lambda f: (os.path.splitext(f)[1], 
                                                  os.path.getmtime(f)))
        
        # Get actual result
        result = sort_func(test_dir)
        
        # Calculate fitness with graduated scoring
        fitness = 0.0
        
        # Base fitness for producing any valid result
        if isinstance(result, list):
            fitness += 10.0
            
            # Reward functions that return the correct data type
            if all(isinstance(f, str) for f in result):
                fitness += 5.0
                
                # Calculate Spearman rank correlation
                expected_ranks = {f: i for i, f in enumerate(expected)}
                actual_ranks = {f: i for i, f in enumerate(result)}
                
                # Get common files between the two lists
                common_files = set(expected_ranks.keys()) & set(actual_ranks.keys())
                
                if common_files:
                    # Calculate rank correlation
                    expected_rank_list = [expected_ranks[f] for f in common_files]
                    actual_rank_list = [actual_ranks[f] for f in common_files]
                    
                    # Simple rank correlation calculation
                    n = len(common_files)
                    if n > 1:
                        d_squared = sum((x - y) ** 2 for x, y in zip(expected_rank_list, actual_rank_list))
                        correlation = 1 - (6 * d_squared) / (n * (n**2 - 1))
                        
                        # Scale correlation from [-1,1] to [0,80]
                        fitness += 40 * (correlation + 1)
                        
                        # Perfect sorting gets a bonus
                        if correlation > 0.99:
                            fitness += 20.0
        
        return fitness
        
    except Exception as e:
        print(f"Error in evaluate_file_sorting_function: {e}")
        return 0.0

def main():
    """Run the evolution process."""
    # Initialize the evolution engine with enhanced configuration
    engine = ExonEvolutionEngine({
        'population_size': 50,  # Increased population size
        'invalid_retention_rate': 0.1,
        'granularity_weights': {
            'function': 0.4,  # Reduced function-level mutations
            'block': 0.3,     # Increased block-level mutations
            'statement': 0.2, # Increased statement-level mutations
            'node': 0.1       # Increased node-level mutations
        }
    })
    
    # Create initial population
    population = INITIAL_POPULATION * 10  # Duplicate to get 50 individuals
    
    # Run evolution
    print("Starting evolution for file sorting...")
    engine.population = population
    
    # Evolve for extension sorting first
    print("\nEvolving extension-based sorting...")
    best_solution = engine.evolve(
        lambda ast_node: evaluate_file_sorting_function(ast_node, 'extension'),
        50  # Increased generations
    )
    
    # Print results
    print("\nEvolution complete!")
    print(f"Best fitness achieved: {engine.best_fitness:.2f}")
    
    # Print the best solution
    if best_solution:
        print("\nBest solution:")
        print(astor.to_source(best_solution))
        
        # Test the best solution
        print("\nTesting best solution:")
        namespace = {}
        exec(astor.to_source(best_solution), namespace)
        sort_func = namespace['sort_files']
        
        test_dir = "test_files"
        files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
        print(f"Input files: {files}")
        print(f"Sorted files: {sort_func(test_dir)}")

if __name__ == '__main__':
    main() 