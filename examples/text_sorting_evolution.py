import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trisolaris.tasks.text_file_sorting import TextFileSortingTask
from trisolaris.core.evolutionary_engine import EvolutionaryEngine
from trisolaris.utils.data_generator import generate_text_files

def main():
    # Create a temporary directory for the text files
    data_dir = Path("temp_text_files")
    data_dir.mkdir(exist_ok=True)
    
    # Generate sample text files
    print("Generating sample text files...")
    ground_truth = generate_text_files(data_dir)
    
    # Create the text sorting task
    task = TextFileSortingTask(str(data_dir), ground_truth)
    
    # Initialize the evolutionary engine
    engine = EvolutionaryEngine(
        task=task,
        population_size=50,
        max_generations=100,
        initial_mutation_rate=0.1,
        initial_selection_pressure=0.5
    )
    
    # Run the evolution
    print("\nStarting evolution...")
    history = engine.run()
    
    # Print results
    print("\nEvolution complete!")
    print(f"Best fitness achieved: {max(history['best_fitness']):.4f}")
    print(f"Final average fitness: {history['avg_fitness'][-1]:.4f}")
    print(f"Final population diversity: {history['diversity'][-1]:.4f}")
    
    # Get the best program
    best_program = engine.population_manager.program_population[
        max(range(len(engine.population_manager.program_population)),
            key=lambda i: task.evaluate_fitness(engine.population_manager.program_population[i]))
    ]
    
    print("\nBest program found:")
    print(best_program.to_source())
    
    # Clean up
    print("\nCleaning up temporary files...")
    for file in data_dir.glob("*"):
        file.unlink()
    data_dir.rmdir()

if __name__ == "__main__":
    main() 