"""
Test script demonstrating the scaffolded evolution approach with progressive task complexity.
"""

import ast
import astor
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from trisolaris.operators import ExonMutator, CodeValidator, ExonEvolutionEngine
from trisolaris.utils.ast_utils import get_function_nodes
import logging

logger = logging.getLogger(__name__)

# Initial population with functional building blocks for extension sorting
INITIAL_EXTENSION_POPULATION = [
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
    
    # Extension-based sorting with error handling
    ast.parse("""
def sort_files(directory):
    try:
        files = os.listdir(directory)
        return sorted(files, key=lambda f: os.path.splitext(f)[1])
    except Exception as e:
        print(f"Error: {e}")
        return []
    """),
    
    # Extension-based sorting with filtering
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    # Filter out directories
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    return sorted(files, key=lambda f: os.path.splitext(f)[1])
    """),
    
    # Extension-based sorting with grouping
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    # Group by extension
    groups = {}
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext not in groups:
            groups[ext] = []
        groups[ext].append(f)
    # Sort each group
    for ext in groups:
        groups[ext].sort()
    return [f for ext in sorted(groups.keys()) for f in groups[ext]]
    """)
]

# Initial population for date-based sorting
INITIAL_DATE_POPULATION = [
    # Basic date sorting
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    """),
    
    # Date sorting with error handling
    ast.parse("""
def sort_files(directory):
    try:
        files = os.listdir(directory)
        return sorted(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    except Exception as e:
        print(f"Error: {e}")
        return []
    """),
    
    # Date sorting with filtering
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    # Filter out directories
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    """),
    
    # Date sorting with grouping
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    # Group by date (day)
    groups = {}
    for f in files:
        mtime = os.path.getmtime(os.path.join(directory, f))
        day = int(mtime / (24 * 3600))
        if day not in groups:
            groups[day] = []
        groups[day].append(f)
    # Sort each group
    for day in groups:
        groups[day].sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return [f for day in sorted(groups.keys()) for f in groups[day]]
    """),
    
    # Date sorting with metadata
    ast.parse("""
def sort_files(directory):
    files = os.listdir(directory)
    # Get file metadata
    file_info = []
    for f in files:
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            stat = os.stat(path)
            file_info.append({
                'name': f,
                'mtime': stat.st_mtime,
                'size': stat.st_size
            })
    # Sort by modification time
    file_info.sort(key=lambda x: x['mtime'])
    return [f['name'] for f in file_info]
    """)
]

def evaluate_file_sorting_function(ast_node: ast.AST) -> float:
    """
    Evaluate a file sorting function's fitness.
    
    Args:
        ast_node: AST of the file sorting function
        
    Returns:
        Fitness score based on correctness
    """
    try:
        # Validate and repair AST if necessary
        validator = CodeValidator()
        if not validator.is_valid_syntax(ast_node):
            logger.warning("Invalid AST detected, attempting repair...")
            ast_node = validator.repair(ast_node)
            if not validator.is_valid_syntax(ast_node):
                logger.error("AST repair failed, skipping evaluation.")
                return 0.0
        
        # Convert AST to source code using ast.unparse
        source = ast.unparse(ast_node)
        
        # Create a namespace for execution
        namespace = {}
        exec(source, namespace)
        
        # Get the sorting function
        sort_func = namespace['sort_files_by_date']
        
        # Test cases: list of filenames in a directory
        test_dir = "GAURAV"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            # Create some dummy files with different modification times
            for i in range(5):
                with open(os.path.join(test_dir, f"file{i}.txt"), "w") as f:
                    f.write(f"dummy content {i}")
                # Set modification time to be different for each file
                os.utime(os.path.join(test_dir, f"file{i}.txt"), (0, i * 1000))
        
        files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
        expected = sorted(files, key=os.path.getmtime)
        result = sort_func(files)
        
        # Score based on correctness
        score = 1.0 if result == expected else 0.0
        return score
        
    except Exception as e:
        logger.error(f"Error in evaluate_file_sorting_function: {e}")
        return 0.0

def evolve_sorting_function(initial_population: List[ast.AST], 
                          task_type: str,
                          population_size: int = 50,
                          generations: int = 50) -> Tuple[ast.AST, float]:
    """
    Evolve a file sorting function for a specific task type.
    
    Args:
        initial_population: List of initial ASTs
        task_type: Type of sorting task ('extension', 'date', 'combined')
        population_size: Size of the population
        generations: Number of generations to evolve
        
    Returns:
        Tuple of (best solution AST, best fitness score)
    """
    # Initialize the evolution engine
    engine = ExonEvolutionEngine({
        'population_size': population_size,
        'invalid_retention_rate': 0.1,
        'granularity_weights': {
            'function': 0.4,  # Reduced function-level mutations
            'block': 0.3,     # Increased block-level mutations
            'statement': 0.2, # Increased statement-level mutations
            'node': 0.1       # Increased node-level mutations
        }
    })
    
    # Create initial population
    population = initial_population * (population_size // len(initial_population))
    if len(population) < population_size:
        population.extend(initial_population[:population_size - len(population)])
    
    # Run evolution
    print(f"\nEvolving {task_type}-based sorting...")
    engine.population = population
    best_solution = engine.evolve(
        lambda ast_node: evaluate_file_sorting_function(ast_node),
        generations
    )
    
    return best_solution, engine.best_fitness

def main():
    """Run the scaffolded evolution process."""
    # Step 1: Evolve extension-based sorting
    print("Step 1: Evolving extension-based sorting...")
    ext_solution, ext_fitness = evolve_sorting_function(
        INITIAL_EXTENSION_POPULATION,
        'extension',
        population_size=50,
        generations=50
    )
    
    print(f"\nExtension sorting evolution complete!")
    print(f"Best fitness achieved: {ext_fitness:.2f}")
    
    if ext_solution:
        print("\nBest extension sorting solution:")
        print(astor.to_source(ext_solution))
    
    # Step 2: Evolve date-based sorting
    print("\nStep 2: Evolving date-based sorting...")
    date_solution, date_fitness = evolve_sorting_function(
        INITIAL_DATE_POPULATION,
        'date',
        population_size=50,
        generations=50
    )
    
    print(f"\nDate sorting evolution complete!")
    print(f"Best fitness achieved: {date_fitness:.2f}")
    
    if date_solution:
        print("\nBest date sorting solution:")
        print(astor.to_source(date_solution))
    
    # Step 3: Combine the best solutions
    print("\nStep 3: Combining best solutions...")
    combined_population = [ext_solution, date_solution] if ext_solution and date_solution else []
    if combined_population:
        combined_solution, combined_fitness = evolve_sorting_function(
            combined_population,
            'combined',
            population_size=50,
            generations=50
        )
        
        print(f"\nCombined sorting evolution complete!")
        print(f"Best fitness achieved: {combined_fitness:.2f}")
        
        if combined_solution:
            print("\nBest combined solution:")
            print(astor.to_source(combined_solution))
            
            # Test the final solution
            print("\nTesting final solution:")
            namespace = {}
            exec(astor.to_source(combined_solution), namespace)
            sort_func = namespace['sort_files']
            
            test_dir = "test_files"
            files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
            print(f"Input files: {files}")
            print(f"Sorted files: {sort_func(test_dir)}")

if __name__ == '__main__':
    main() 