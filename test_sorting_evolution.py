"""
Test script demonstrating the exon-like mutation system by evolving a sorting function.
"""

import ast
import astor
from trisolaris.operators import ExonMutator, CodeValidator, ExonEvolutionEngine
from trisolaris.utils.ast_utils import get_function_nodes
import os

# Initial population of sorting functions
INITIAL_POPULATION = [
    ast.parse("""
def sort_list(lst):
    return sorted(lst)
    """),
    ast.parse("""
def sort_list(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst
    """),
    ast.parse("""
def sort_list(lst):
    if not lst:
        return []
    pivot = lst[0]
    left = [x for x in lst[1:] if x <= pivot]
    right = [x for x in lst[1:] if x > pivot]
    return sort_list(left) + [pivot] + sort_list(right)
    """)
]

# New initial population for file sorting by modification date
INITIAL_FILE_POPULATION = [
    ast.parse("""
def sort_files_by_date(files):
    return sorted(files, key=lambda f: os.path.getmtime(f))
    """),
    ast.parse("""
def sort_files_by_date(files):
    return sorted(files, key=os.path.getmtime)
    """),
    ast.parse("""
def sort_files_by_date(files):
    return sorted(files, key=lambda f: os.stat(f).st_mtime)
    """)
]

def evaluate_sorting_function(ast_node: ast.AST) -> float:
    """
    Evaluate a sorting function's fitness.
    
    Args:
        ast_node: AST of the sorting function
        
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
        sort_func = namespace['sort_list']
        
        # Test cases
        test_cases = [
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # Already sorted
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),  # Reverse sorted
            ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),  # Duplicates
            ([], []),                             # Empty list
            ([1], [1]),                          # Single element
        ]
        
        # Score based on correctness
        score = 0
        for input_list, expected in test_cases:
            try:
                result = sort_func(input_list.copy())
                if result == expected:
                    score += 1
            except Exception:
                continue
        
        # Normalize score
        return score / len(test_cases)
        
    except Exception:
        return 0.0

def evaluate_file_sorting_function(ast_node: ast.AST) -> float:
    """
    Evaluate a file sorting function's fitness.
    
    Args:
        ast_node: AST of the file sorting function
        
    Returns:
        Fitness score based on correctness
    """
    try:
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
        print(f"Error in evaluate_file_sorting_function: {e}")
        return 0.0

def main():
    """Run the evolution process."""
    # Initialize the evolution engine
    engine = ExonEvolutionEngine({
        'population_size': 10,
        'invalid_retention_rate': 0.1,
        'granularity_weights': {
            'function': 0.6,  # Favor function-level mutations
            'block': 0.3,
            'statement': 0.1,
            'node': 0.0
        }
    })
    
    # Create initial population for file sorting
    population = INITIAL_FILE_POPULATION * 4  # Duplicate to get 12 individuals
    
    # Run evolution
    print("Starting evolution for file sorting by modification date...")
    engine.population = population
    best_solution = engine.evolve(
        evaluate_file_sorting_function,
        10
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
        sort_func = namespace['sort_files_by_date']
        
        test_dir = "GAURAV"
        files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
        print(f"Input files: {files}")
        print(f"Sorted files: {sort_func(files)}")

if __name__ == '__main__':
    main() 