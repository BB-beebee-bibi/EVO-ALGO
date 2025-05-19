#!/usr/bin/env python3
"""
Simple Evolutionary Algorithm Demonstration

This script demonstrates the core functionality of the evolutionary algorithm,
showing how it produces viable offspring through mutation and crossover operations.
It provides a simple, interactive interface for users to experiment with the system.
"""

import os
import sys
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from trisolaris.core.program_representation import ProgramAST
from trisolaris.utils.data_generator import generate_text_files, SAMPLE_TEXTS

class SimpleEvolutionDemo:
    """
    A simplified demonstration of the evolutionary algorithm.
    Focuses on showing how programs evolve through mutation and crossover.
    """
    
    def __init__(self, population_size: int = 10, generations: int = 10):
        """Initialize the demo with configurable parameters."""
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Create initial population
        self._initialize_population()
        
        # Track fitness scores
        self.fitness_scores = []
        
    def _initialize_population(self):
        """Create an initial population of programs."""
        self.population = []
        for _ in range(self.population_size):
            # Create a simple program that attempts to sort files
            program = ProgramAST(source="""
def sort_files(file_list):
    # Initial implementation just returns a dictionary with random categories
    result = {}
    categories = ['bible', 'shakespeare', 'war_and_peace', 'e40', 'gurbani']
    for file in file_list:
        result[file] = random.choice(categories)
    return result
""")
            self.population.append(program)
    
    def _evaluate_fitness(self, program: ProgramAST, test_files: Dict[str, str]) -> float:
        """
        Evaluate the fitness of a program based on how well it sorts files.
        Returns a score between 0 and 1.
        """
        try:
            # Convert AST to executable code
            source_code = program.to_source()
            namespace = {"random": random}
            exec(source_code, namespace)
            
            # Get the sort_files function
            sort_files = namespace.get('sort_files')
            if not sort_files:
                return 0.0
            
            # Run the program on our file list
            file_list = list(test_files.keys())
            result = sort_files(file_list)
            
            # Calculate accuracy
            correct = 0
            total = len(test_files)
            
            for filename, predicted_category in result.items():
                if filename in test_files and predicted_category == test_files[filename]:
                    correct += 1
            
            # Return accuracy as fitness
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error evaluating program: {e}")
            return 0.0
    
    def _select_parents(self, fitness_scores: List[float]) -> List[ProgramAST]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = max(2, min(3, len(self.population) // 2))
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            parents.append(self.population[winner_idx])
        
        return parents
    
    def _create_next_generation(self, parents: List[ProgramAST], mutation_rate: float = 0.3):
        """Create the next generation through crossover and mutation."""
        next_generation = []
        
        # Elitism: Keep the best individual
        best_idx = max(range(len(self.population)), 
                      key=lambda i: self.fitness_scores[i])
        next_generation.append(self.population[best_idx])
        
        # Create offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            if len(parents) >= 2 and random.random() < 0.7:  # Crossover rate
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Perform crossover
                child1, child2 = ProgramAST.crossover(parent1, parent2)
                
                # Mutate children
                if random.random() < mutation_rate:
                    child1 = child1.mutate(mutation_rate)
                if random.random() < mutation_rate:
                    child2 = child2.mutate(mutation_rate)
                
                next_generation.extend([child1, child2])
            else:
                # Just mutation
                parent = random.choice(parents)
                child = parent.mutate(mutation_rate)
                next_generation.append(child)
        
        # Ensure we don't exceed population size
        self.population = next_generation[:self.population_size]
    
    def run_evolution(self, test_files: Dict[str, str]):
        """Run the evolutionary process for the specified number of generations."""
        print(f"Starting evolution with population size {self.population_size} for {self.generations} generations")
        
        for generation in range(self.generations):
            # Evaluate fitness
            self.fitness_scores = [self._evaluate_fitness(program, test_files) for program in self.population]
            
            # Track statistics
            avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
            best_fitness = max(self.fitness_scores)
            self.avg_fitness_history.append(avg_fitness)
            self.best_fitness_history.append(best_fitness)
            
            # Print progress
            print(f"Generation {generation+1}/{self.generations}: "
                  f"Best fitness = {best_fitness:.4f}, "
                  f"Avg fitness = {avg_fitness:.4f}")
            
            # Select parents
            parents = self._select_parents(self.fitness_scores)
            
            # Create next generation
            self._create_next_generation(parents)
        
        # Final evaluation
        self.fitness_scores = [self._evaluate_fitness(program, test_files) for program in self.population]
        best_idx = max(range(len(self.population)), key=lambda i: self.fitness_scores[i])
        best_program = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]
        
        print("\nEvolution complete!")
        print(f"Best fitness achieved: {best_fitness:.4f}")
        
        return best_program, best_fitness
    
    def plot_fitness_history(self):
        """Plot the fitness history over generations."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_fitness_history) + 1), self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(range(1, len(self.avg_fitness_history) + 1), self.avg_fitness_history, 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('fitness_evolution.png')
        print("Fitness evolution plot saved as 'fitness_evolution.png'")
        
        # Show the plot
        plt.show()

def generate_test_data():
    """Generate test data for the evolution demo."""
    # Create a temporary directory for the text files
    data_dir = Path("temp_text_files")
    data_dir.mkdir(exist_ok=True)
    
    # Generate sample text files
    print("Generating sample text files...")
    ground_truth = generate_text_files(data_dir)
    
    return data_dir, ground_truth

def cleanup_test_data(data_dir: Path):
    """Clean up the test data files."""
    print("\nCleaning up temporary files...")
    for file in data_dir.glob("*"):
        file.unlink()
    data_dir.rmdir()

def interactive_demo():
    """Run an interactive demonstration of the evolutionary algorithm."""
    print("=" * 80)
    print("EVOLUTIONARY ALGORITHM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how the evolutionary algorithm produces viable offspring")
    print("through mutation and crossover operations.\n")
    
    # Get user parameters
    try:
        pop_size = int(input("Enter population size (default 10): ") or "10")
        generations = int(input("Enter number of generations (default 10): ") or "10")
        show_plot = input("Show fitness plot at the end? (y/n, default y): ").lower() != 'n'
    except ValueError:
        print("Invalid input. Using default values.")
        pop_size = 10
        generations = 10
        show_plot = True
    
    # Generate test data
    data_dir, test_files = generate_test_data()
    
    # Create and run the demo
    demo = SimpleEvolutionDemo(population_size=pop_size, generations=generations)
    
    # Time the evolution process
    start_time = time.time()
    best_program, best_fitness = demo.run_evolution(test_files)
    elapsed_time = time.time() - start_time
    
    # Show results
    print(f"\nEvolution completed in {elapsed_time:.2f} seconds")
    print("\nBest program found:")
    print("-" * 40)
    print(best_program.to_source())
    print("-" * 40)
    
    # Test the best program
    print("\nTesting the best program on the data:")
    source_code = best_program.to_source()
    namespace = {"random": random}
    exec(source_code, namespace)
    sort_files = namespace.get('sort_files')
    
    if sort_files:
        file_list = list(test_files.keys())
        result = sort_files(file_list)
        
        print("\nSample classifications:")
        for i, (filename, category) in enumerate(result.items()):
            correct = "✓" if category == test_files[filename] else "✗"
            print(f"{filename}: Predicted '{category}', Actual '{test_files[filename]}' {correct}")
            if i >= 4:  # Show only first 5 examples
                print(f"... and {len(result) - 5} more")
                break
        
        # Calculate and show accuracy
        correct_count = sum(1 for f, c in result.items() if c == test_files[f])
        accuracy = correct_count / len(test_files) if test_files else 0
        print(f"\nOverall accuracy: {accuracy:.2%} ({correct_count}/{len(test_files)} correct)")
    
    # Show fitness plot
    if show_plot:
        demo.plot_fitness_history()
    
    # Clean up
    cleanup_test_data(data_dir)
    
    print("\nDemonstration complete!")

def show_example_evolved_program():
    """Show an example of what an evolved program might look like."""
    print("\nExample of an evolved program that might be produced:")
    print("-" * 40)
    example_program = """
def sort_files(file_list):
    result = {}
    for filename in file_list:
        # Look for keywords in the filename
        if 'bible' in filename.lower():
            result[filename] = 'bible'
        elif 'shake' in filename.lower():
            result[filename] = 'shakespeare'
        elif 'war' in filename.lower() or 'peace' in filename.lower():
            result[filename] = 'war_and_peace'
        elif 'e40' in filename.lower() or 'yay' in filename.lower():
            result[filename] = 'e40'
        elif 'guru' in filename.lower() or 'sikh' in filename.lower():
            result[filename] = 'gurbani'
        else:
            # If no keywords found, try to analyze content
            try:
                with open(filename, 'r') as f:
                    content = f.read().lower()
                    
                if any(word in content for word in ['god', 'jesus', 'lord', 'heaven']):
                    result[filename] = 'bible'
                elif any(word in content for word in ['thou', 'thee', 'thy', 'hath']):
                    result[filename] = 'shakespeare'
                elif any(word in content for word in ['war', 'peace', 'russia', 'napoleon']):
                    result[filename] = 'war_and_peace'
                elif any(word in content for word in ['yay', 'fo', 'sizzurp', 'hyphy']):
                    result[filename] = 'e40'
                elif any(word in content for word in ['waheguru', 'guru', 'sikh', 'gurbani']):
                    result[filename] = 'gurbani'
                else:
                    # Default if no patterns match
                    result[filename] = 'unknown'
            except:
                # If file can't be read, make a random guess
                result[filename] = random.choice(['bible', 'shakespeare', 'war_and_peace', 'e40', 'gurbani'])
    
    return result
"""
    print(example_program)
    print("-" * 40)
    print("Note: This is just an example of what might evolve. The actual evolved programs")
    print("will vary based on random factors and the specific evolutionary process.")

if __name__ == "__main__":
    show_example_evolved_program()
    interactive_demo()