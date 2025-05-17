import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trisolaris.core.program_representation import ProgramAST
from trisolaris.tasks.text_file_sorting import TextFileSortingTask
from trisolaris.utils.data_generator import generate_text_files

def run_mutation_smoke_test(
    population_size: int = 50,
    generations: int = 10,
    mutation_rate: float = 0.1,
    save_plot: bool = True
):
    """Run a smoke test of the mutation operators and visualize fitness spread."""
    
    # Create a temporary directory for the text files
    data_dir = Path("temp_text_files")
    data_dir.mkdir(exist_ok=True)
    
    # Generate sample text files
    print("Generating sample text files...")
    ground_truth = generate_text_files(data_dir)
    
    # Create the text sorting task
    task = TextFileSortingTask(str(data_dir), ground_truth)
    
    # Initialize population
    population = [ProgramAST() for _ in range(population_size)]
    
    # Track fitness history
    fitness_history = []
    best_fitness_history = []
    
    # Evolution loop
    print("\nStarting evolution...")
    for generation in range(generations):
        # Evaluate current population
        fitness_scores = [task.evaluate_fitness(program) for program in population]
        
        # Record statistics
        fitness_history.append(fitness_scores)
        best_fitness_history.append(max(fitness_scores))
        
        print(f"Generation {generation + 1}/{generations}")
        print(f"Best fitness: {max(fitness_scores):.4f}")
        print(f"Average fitness: {np.mean(fitness_scores):.4f}")
        print(f"Fitness std dev: {np.std(fitness_scores):.4f}\n")
        
        # Create next generation through mutation
        new_population = []
        for program in population:
            mutated = program.mutate(mutation_rate)
            new_population.append(mutated)
        population = new_population
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot fitness distribution over generations
    plt.subplot(1, 2, 1)
    plt.boxplot(fitness_history, labels=[f"Gen {i+1}" for i in range(generations)])
    plt.title("Fitness Distribution Over Generations")
    plt.ylabel("Fitness")
    plt.xticks(rotation=45)
    
    # Plot best fitness progression
    plt.subplot(1, 2, 2)
    plt.plot(best_fitness_history, 'b-', label='Best Fitness')
    plt.title("Best Fitness Progression")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig("mutation_smoke_test_results.png")
        print("\nResults plot saved as 'mutation_smoke_test_results.png'")
    else:
        plt.show()
    
    # Clean up
    print("\nCleaning up temporary files...")
    for file in data_dir.glob("*"):
        file.unlink()
    data_dir.rmdir()

if __name__ == "__main__":
    run_mutation_smoke_test() 