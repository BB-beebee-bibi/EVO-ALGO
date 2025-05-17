import os
from typing import List, Dict, Any
import ast
import astor
from ..core.program_representation import ProgramAST

class TextFileSortingTask:
    """
    Evolutionary task for sorting/classifying text files by their true content.
    Designed for extensibility to other file types in the future.
    """
    def __init__(self, data_dir: str, ground_truth: Dict[str, str]):
        self.data_dir = data_dir
        self.ground_truth = ground_truth
        self.file_list = list(ground_truth.keys())
        
        # Compile ground truth categories for quick lookup
        self.categories = set(ground_truth.values())
        
        # Common keywords for each category (can be expanded)
        self.category_keywords = {
            'bible': ['god', 'jesus', 'lord', 'heaven', 'sin', 'faith'],
            'shakespeare': ['thou', 'thee', 'thy', 'hath', 'doth', 'forsooth'],
            'war_and_peace': ['war', 'peace', 'russia', 'napoleon', 'battle'],
            'e40': ['yay', 'fo', 'sizzurp', 'hyphy', 'thizz'],
            'gurbani': ['waheguru', 'guru', 'sikh', 'sikhi', 'gurbani']
        }

    def initialize_population(self, pop_size: int) -> List[ProgramAST]:
        """Initialize a population of candidate sorting programs."""
        return [ProgramAST() for _ in range(pop_size)]

    def evaluate_fitness(self, candidate_program: ProgramAST) -> float:
        """
        Evaluate how well a candidate sorts files by content.
        Returns a fitness score between 0 and 1.
        """
        try:
            # Convert AST to executable code
            source_code = candidate_program.to_source()
            namespace = {}
            exec(source_code, namespace)
            
            # Get the sort_files function
            sort_files = namespace['sort_files']
            
            # Run the program on our file list
            result = sort_files(self.file_list)
            
            # Calculate accuracy
            correct = 0
            total = len(self.ground_truth)
            
            for filename, predicted_category in result.items():
                if filename in self.ground_truth:
                    if predicted_category == self.ground_truth[filename]:
                        correct += 1
            
            # Base accuracy score
            accuracy = correct / total
            
            # Additional metrics for better fitness evaluation
            category_distribution = self._calculate_category_distribution(result)
            distribution_score = self._evaluate_distribution(category_distribution)
            
            # Combine scores (70% accuracy, 30% distribution)
            final_score = 0.7 * accuracy + 0.3 * distribution_score
            
            return final_score
            
        except Exception as e:
            # Penalize programs that fail to execute
            return 0.0

    def _calculate_category_distribution(self, result: Dict[str, str]) -> Dict[str, float]:
        """Calculate the distribution of categories in the result."""
        total = len(result)
        if total == 0:
            return {category: 0.0 for category in self.categories}
        
        distribution = {}
        for category in self.categories:
            count = sum(1 for pred in result.values() if pred == category)
            distribution[category] = count / total
        
        return distribution

    def _evaluate_distribution(self, distribution: Dict[str, float]) -> float:
        """
        Evaluate how well the distribution matches the ground truth.
        Returns a score between 0 and 1.
        """
        # Calculate ground truth distribution
        total = len(self.ground_truth)
        ground_truth_dist = {
            category: sum(1 for cat in self.ground_truth.values() if cat == category) / total
            for category in self.categories
        }
        
        # Calculate distribution similarity (using cosine similarity)
        dot_product = sum(distribution[cat] * ground_truth_dist[cat] for cat in self.categories)
        magnitude1 = sum(x * x for x in distribution.values()) ** 0.5
        magnitude2 = sum(x * x for x in ground_truth_dist.values()) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def read_file_content(self, filename: str) -> str:
        """Read the content of a file."""
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            return f.read()

    def extend_to_filetype(self, filetype: str):
        """Stub for future extension to new file types (e.g., CSV, DataFrame)."""
        pass
