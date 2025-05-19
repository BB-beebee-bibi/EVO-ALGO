#!/usr/bin/env python3
"""
Simple Evolutionary Algorithm Demonstration

This script demonstrates a basic evolutionary algorithm that evolves text classifiers.
It shows how the algorithm produces viable offspring through mutation and crossover operations.
"""

import os
import sys
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable

# Sample text snippets for each category
SAMPLE_TEXTS = {
    'bible': [
        "In the beginning God created the heaven and the earth.",
        "And God said, Let there be light: and there was light.",
        "And God saw the light, that it was good: and God divided the light from the darkness."
    ],
    'shakespeare': [
        "To be, or not to be, that is the question.",
        "All the world's a stage, and all the men and women merely players.",
        "What's in a name? That which we call a rose by any other name would smell as sweet."
    ],
    'war_and_peace': [
        "Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes.",
        "The war was not a game of cards, but a serious business.",
        "The strongest of all warriors are these two — Time and Patience."
    ],
    'e40': [
        "Tell me when ta GO",
        "It's all good.",
        "GO STUPID"
    ],
    'gurbani': [
        "Ik Onkar, Sat Naam, Karta Purakh.",
        "Waheguru Ji Ka Khalsa, Waheguru Ji Ki Fateh.",
        "Guru Ram Das Ji, the fourth Guru, spread the message of love and equality."
    ]
}

# Keywords for each category
KEYWORDS = {
    'bible': ['god', 'jesus', 'lord', 'heaven', 'sin', 'faith'],
    'shakespeare': ['thou', 'thee', 'thy', 'hath', 'doth', 'forsooth'],
    'war_and_peace': ['war', 'peace', 'russia', 'napoleon', 'battle'],
    'e40': ['yay', 'fo', 'sizzurp', 'hyphy', 'thizz'],
    'gurbani': ['waheguru', 'guru', 'sikh', 'sikhi', 'gurbani']
}

class TextClassifier:
    """A simple text classifier that can be evolved."""
    
    def __init__(self, keywords=None):
        """Initialize with optional keywords for each category."""
        self.keywords = keywords or {
            'bible': random.sample(KEYWORDS['bible'], k=min(2, len(KEYWORDS['bible']))),
            'shakespeare': random.sample(KEYWORDS['shakespeare'], k=min(2, len(KEYWORDS['shakespeare']))),
            'war_and_peace': random.sample(KEYWORDS['war_and_peace'], k=min(2, len(KEYWORDS['war_and_peace']))),
            'e40': random.sample(KEYWORDS['e40'], k=min(2, len(KEYWORDS['e40']))),
            'gurbani': random.sample(KEYWORDS['gurbani'], k=min(2, len(KEYWORDS['gurbani'])))
        }
        
        # Random weights for each category (how much to prioritize each category)
        self.weights = {
            'bible': random.random(),
            'shakespeare': random.random(),
            'war_and_peace': random.random(),
            'e40': random.random(),
            'gurbani': random.random()
        }
    
    def classify(self, text: str) -> str:
        """Classify a text into one of the predefined categories."""
        text = text.lower()
        scores = {}
        
        for category, keywords in self.keywords.items():
            # Count keyword occurrences
            score = sum(text.count(keyword) for keyword in keywords)
            # Apply weight
            scores[category] = score * self.weights[category]
        
        # Return the category with the highest score, or a random one if all scores are 0
        max_score = max(scores.values())
        if max_score > 0:
            # Get all categories with the max score (in case of ties)
            top_categories = [cat for cat, score in scores.items() if score == max_score]
            return random.choice(top_categories)
        else:
            return random.choice(list(self.keywords.keys()))
    
    def classify_files(self, files_content: Dict[str, str]) -> Dict[str, str]:
        """Classify multiple files and return a mapping of filename to category."""
        result = {}
        for filename, content in files_content.items():
            result[filename] = self.classify(content)
        return result
    
    def mutate(self, mutation_rate: float = 0.3) -> 'TextClassifier':
        """Create a mutated copy of this classifier."""
        # Create a copy of the current classifier
        new_classifier = TextClassifier(keywords={k: list(v) for k, v in self.keywords.items()})
        new_classifier.weights = dict(self.weights)
        
        # Mutate keywords
        if random.random() < mutation_rate:
            # Pick a random category
            category = random.choice(list(self.keywords.keys()))
            
            # Mutation type: add, remove, or replace a keyword
            mutation_type = random.choice(['add', 'remove', 'replace'])
            
            if mutation_type == 'add' and len(new_classifier.keywords[category]) < len(KEYWORDS[category]):
                # Add a new keyword from the master list that's not already in use
                available = [k for k in KEYWORDS[category] if k not in new_classifier.keywords[category]]
                if available:
                    new_classifier.keywords[category].append(random.choice(available))
            
            elif mutation_type == 'remove' and len(new_classifier.keywords[category]) > 1:
                # Remove a random keyword
                new_classifier.keywords[category].remove(random.choice(new_classifier.keywords[category]))
            
            elif mutation_type == 'replace' and KEYWORDS[category]:
                # Replace a keyword with another from the master list
                if new_classifier.keywords[category]:  # Make sure there's at least one keyword to replace
                    idx = random.randrange(len(new_classifier.keywords[category]))
                    available = [k for k in KEYWORDS[category] if k not in new_classifier.keywords[category]]
                    if available:
                        new_classifier.keywords[category][idx] = random.choice(available)
        
        # Mutate weights
        for category in self.weights:
            if random.random() < mutation_rate:
                # Adjust weight by up to ±50%
                adjustment = (random.random() - 0.5)  # -0.25 to +0.25
                new_classifier.weights[category] *= (1 + adjustment)
                # Ensure weight stays positive
                new_classifier.weights[category] = max(0.1, new_classifier.weights[category])
        
        return new_classifier
    
    @classmethod
    def crossover(cls, parent1: 'TextClassifier', parent2: 'TextClassifier') -> tuple['TextClassifier', 'TextClassifier']:
        """Perform crossover between two parent classifiers."""
        # Create two new children
        child1 = cls()
        child2 = cls()
        
        # Crossover keywords
        categories = list(parent1.keywords.keys())
        for category in categories:
            # Randomly decide which parent's keywords to use for each category
            if random.random() < 0.5:
                child1.keywords[category] = list(parent1.keywords[category])
                child2.keywords[category] = list(parent2.keywords[category])
            else:
                child1.keywords[category] = list(parent2.keywords[category])
                child2.keywords[category] = list(parent1.keywords[category])
        
        # Crossover weights
        for category in categories:
            # Randomly decide which parent's weight to use for each category
            if random.random() < 0.5:
                child1.weights[category] = parent1.weights[category]
                child2.weights[category] = parent2.weights[category]
            else:
                child1.weights[category] = parent2.weights[category]
                child2.weights[category] = parent1.weights[category]
        
        return child1, child2
    
    def __str__(self) -> str:
        """Return a string representation of the classifier."""
        result = "TextClassifier:\n"
        for category, keywords in self.keywords.items():
            result += f"  {category} (weight={self.weights[category]:.2f}): {', '.join(keywords)}\n"
        return result


class EvolutionaryAlgorithm:
    """Simple evolutionary algorithm for text classifiers."""
    
    def __init__(self, population_size: int = 10, generations: int = 10):
        """Initialize the evolutionary algorithm."""
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitness_scores = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self):
        """Create an initial population of classifiers."""
        self.population = [TextClassifier() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, classifier: TextClassifier, test_data: Dict[str, Dict[str, str]]) -> float:
        """
        Evaluate the fitness of a classifier based on how well it classifies texts.
        Returns a score between 0 and 1.
        """
        correct = 0
        total = 0
        
        for filename, content in test_data['content'].items():
            predicted = classifier.classify(content)
            actual = test_data['ground_truth'][filename]
            if predicted == actual:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def select_parents(self) -> List[TextClassifier]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = max(2, min(3, len(self.population) // 2))
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            winner_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
            parents.append(self.population[winner_idx])
        
        return parents
    
    def create_next_generation(self, parents: List[TextClassifier], mutation_rate: float = 0.3):
        """Create the next generation through crossover and mutation."""
        next_generation = []
        
        # Elitism: Keep the best individual
        best_idx = max(range(len(self.population)), key=lambda i: self.fitness_scores[i])
        next_generation.append(self.population[best_idx])
        
        # Create offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            if len(parents) >= 2 and random.random() < 0.7:  # Crossover rate
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Perform crossover
                child1, child2 = TextClassifier.crossover(parent1, parent2)
                
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
    
    def run_evolution(self, test_data: Dict[str, Dict[str, str]]):
        """Run the evolutionary process for the specified number of generations."""
        print(f"Starting evolution with population size {self.population_size} for {self.generations} generations")
        
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            self.fitness_scores = [self.evaluate_fitness(classifier, test_data) for classifier in self.population]
            
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
            parents = self.select_parents()
            
            # Create next generation
            self.create_next_generation(parents)
        
        # Final evaluation
        self.fitness_scores = [self.evaluate_fitness(classifier, test_data) for classifier in self.population]
        best_idx = max(range(len(self.population)), key=lambda i: self.fitness_scores[i])
        best_classifier = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]
        
        print("\nEvolution complete!")
        print(f"Best fitness achieved: {best_fitness:.4f}")
        
        return best_classifier, best_fitness
    
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


def generate_test_data(num_files_per_category: int = 3) -> Dict[str, Dict[str, str]]:
    """Generate test data for the evolution demo."""
    print("Generating sample text files...")
    
    content = {}
    ground_truth = {}
    
    for category, texts in SAMPLE_TEXTS.items():
        for i in range(num_files_per_category):
            # Generate a random filename
            filename = f"file_{random.randint(1000, 9999)}.txt"
            
            # Create content by sampling from the category texts
            file_content = random.choice(texts)
            
            # Store the content and ground truth
            content[filename] = file_content
            ground_truth[filename] = category
    
    return {
        'content': content,
        'ground_truth': ground_truth
    }


def interactive_demo():
    """Run an interactive demonstration of the evolutionary algorithm."""
    print("=" * 80)
    print("EVOLUTIONARY ALGORITHM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how the evolutionary algorithm produces viable offspring")
    print("through mutation and crossover operations.\n")
    
    # Get user parameters
    try:
        pop_size = int(input("Enter population size (default 20): ") or "20")
        generations = int(input("Enter number of generations (default 20): ") or "20")
        show_plot = input("Show fitness plot at the end? (y/n, default y): ").lower() != 'n'
    except ValueError:
        print("Invalid input. Using default values.")
        pop_size = 20
        generations = 20
        show_plot = True
    
    # Generate test data
    test_data = generate_test_data(num_files_per_category=5)
    
    # Create and run the evolutionary algorithm
    algorithm = EvolutionaryAlgorithm(population_size=pop_size, generations=generations)
    
    # Time the evolution process
    start_time = time.time()
    best_classifier, best_fitness = algorithm.run_evolution(test_data)
    elapsed_time = time.time() - start_time
    
    # Show results
    print(f"\nEvolution completed in {elapsed_time:.2f} seconds")
    print("\nBest classifier found:")
    print("-" * 40)
    print(best_classifier)
    print("-" * 40)
    
    # Test the best classifier
    print("\nTesting the best classifier on the data:")
    predictions = {}
    for filename, content in test_data['content'].items():
        predictions[filename] = best_classifier.classify(content)
    
    print("\nSample classifications:")
    correct_count = 0
    for i, (filename, predicted) in enumerate(predictions.items()):
        actual = test_data['ground_truth'][filename]
        correct = "✓" if predicted == actual else "✗"
        if correct == "✓":
            correct_count += 1
        print(f"{filename}: Content: '{test_data['content'][filename][:30]}...' Predicted: '{predicted}', Actual: '{actual}' {correct}")
        if i >= 4:  # Show only first 5 examples
            remaining = len(predictions) - 5
            if remaining > 0:
                print(f"... and {remaining} more")
            break
    
    # Calculate and show accuracy
    accuracy = correct_count / len(test_data['ground_truth']) if test_data['ground_truth'] else 0
    print(f"\nOverall accuracy: {accuracy:.2%} ({correct_count}/{len(test_data['ground_truth'])} correct)")
    
    # Show fitness plot
    if show_plot:
        algorithm.plot_fitness_history()
    
    print("\nDemonstration complete!")


def show_example_evolved_classifier():
    """Show an example of what an evolved classifier might look like."""
    print("\nExample of an evolved classifier that might be produced:")
    print("-" * 40)
    
    # Create a sophisticated classifier
    classifier = TextClassifier()
    
    # Set up keywords that would be effective
    classifier.keywords = {
        'bible': ['god', 'heaven', 'lord', 'faith'],
        'shakespeare': ['thou', 'thee', 'thy', 'hath'],
        'war_and_peace': ['war', 'peace', 'russia', 'napoleon'],
        'e40': ['yay', 'fo', 'hyphy'],
        'gurbani': ['waheguru', 'guru', 'sikh']
    }
    
    # Set up weights that prioritize certain categories
    classifier.weights = {
        'bible': 1.2,
        'shakespeare': 1.5,
        'war_and_peace': 1.0,
        'e40': 1.3,
        'gurbani': 1.1
    }
    
    print(classifier)
    print("-" * 40)
    print("Note: This is just an example of what might evolve. The actual evolved classifiers")
    print("will vary based on random factors and the specific evolutionary process.")


if __name__ == "__main__":
    show_example_evolved_classifier()
    interactive_demo()