"""
Fitness evaluation for evolved programs.
"""
import ast
import logging
from typing import Dict, Any, List, Optional, Callable
from .program_representation import ProgramAST

logger = logging.getLogger(__name__)

class FitnessEvaluator:
    """Evaluates fitness of evolved programs."""
    
    def __init__(self, 
                 test_cases: List[Dict[str, Any]],
                 fitness_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the fitness evaluator.
        
        Args:
            test_cases: List of test cases with inputs and expected outputs
            fitness_weights: Optional weights for different fitness components
        """
        self.test_cases = test_cases
        self.fitness_weights = fitness_weights or {
            'correctness': 0.6,
            'performance': 0.2,
            'complexity': 0.2
        }
        
    def evaluate(self, program: ProgramAST) -> float:
        """
        Evaluate the fitness of a program.
        
        Args:
            program: The program AST to evaluate
            
        Returns:
            Fitness score between 0 and 1
        """
        try:
            # Compile the program
            code = program.to_source()
            namespace = {}
            exec(code, namespace)
            
            # Get the main function
            main_func = namespace.get('main')
            if not main_func:
                return 0.0
                
            # Run test cases
            correctness_score = self._evaluate_correctness(main_func)
            performance_score = self._evaluate_performance(main_func)
            complexity_score = self._evaluate_complexity(program)
            
            # Calculate weighted score
            total_score = (
                correctness_score * self.fitness_weights['correctness'] +
                performance_score * self.fitness_weights['performance'] +
                complexity_score * self.fitness_weights['complexity']
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error evaluating program: {e}")
            return 0.0
            
    def _evaluate_correctness(self, func: Callable) -> float:
        """Evaluate correctness of program outputs."""
        correct = 0
        total = len(self.test_cases)
        
        for test_case in self.test_cases:
            try:
                result = func(test_case['input'])
                if result == test_case['expected']:
                    correct += 1
            except Exception:
                continue
                
        return correct / total if total > 0 else 0.0
        
    def _evaluate_performance(self, func: Callable) -> float:
        """Evaluate performance characteristics."""
        try:
            # For MVP, just check if function completes within time limit
            import time
            start_time = time.time()
            
            # Run on a sample test case
            if self.test_cases:
                func(self.test_cases[0]['input'])
                
            execution_time = time.time() - start_time
            
            # Score based on execution time (lower is better)
            # Cap at 1 second for MVP
            return max(0.0, 1.0 - min(1.0, execution_time))
            
        except Exception:
            return 0.0
            
    def _evaluate_complexity(self, program: ProgramAST) -> float:
        """Evaluate code complexity."""
        try:
            # For MVP, use simple metrics:
            # 1. Number of lines
            # 2. Number of branches
            # 3. Number of loops
            
            source = program.to_source()
            lines = len(source.splitlines())
            
            # Count branches and loops
            branches = 0
            loops = 0
            
            for node in ast.walk(program.tree):
                if isinstance(node, (ast.If, ast.Try)):
                    branches += 1
                elif isinstance(node, (ast.For, ast.While)):
                    loops += 1
                    
            # Calculate complexity score (lower is better)
            # Normalize to 0-1 range
            complexity = (lines / 100.0 + branches / 10.0 + loops / 5.0) / 3.0
            return max(0.0, 1.0 - min(1.0, complexity))
            
        except Exception:
            return 0.0 