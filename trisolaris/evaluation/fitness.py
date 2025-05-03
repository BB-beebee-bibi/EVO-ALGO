"""
Fitness Evaluator for the TRISOLARIS framework.

This module provides the FitnessEvaluator class that evaluates solutions based on
multiple objectives, including alignment with principles, functionality, and resource efficiency.
"""

import time
import tracemalloc
import inspect
import re
import ast
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FitnessEvaluator:
    """
    Evaluates solutions based on multiple objectives.
    
    This class provides a framework for multi-objective fitness evaluation,
    balancing functional correctness, ethical alignment, and resource efficiency.
    """
    
    def __init__(self, 
                ethical_filter=None, 
                weights=None):
        """
        Initialize the Fitness Evaluator.
        
        Args:
            ethical_filter: Optional EthicalBoundaryEnforcer for pre-filtering solutions
            weights: Optional dictionary of weights for different objectives
        """
        # Ethical boundary enforcer (optional)
        self.ethical_filter = ethical_filter
        
        # Default weights for objectives - updated to prioritize alignment
        self.weights = {
            'alignment': 0.6,   # Alignment with principles (increased from 0.5)
            'functionality': 0.25,  # Functional correctness (reduced from 0.3)
            'efficiency': 0.15,   # Resource efficiency (reduced from 0.2)
        }
        
        # Update weights if provided
        if weights:
            for key, value in weights.items():
                if key in self.weights:
                    self.weights[key] = value
            
            # Normalize weights to sum to 1
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Test cases for functional evaluation
        self.test_cases = []
        
        # Resource constraint metrics
        self.resource_constraints = {
            'max_execution_time': 1.0,  # seconds
            'max_memory_usage': 100,    # MB
        }
        
        # Alignment measures with default measures based on universal principles
        self.alignment_measures = [
            {
                'func': self._measure_code_clarity,
                'weight': 0.2,
                'name': "Code clarity and simplicity"
            },
            {
                'func': self._measure_inclusive_language,
                'weight': 0.2,
                'name': "Inclusive and respectful language"
            },
            {
                'func': self._measure_service_orientation,
                'weight': 0.2,
                'name': "Service-oriented design"
            },
            {
                'func': self._measure_mindful_resource_usage,
                'weight': 0.2,
                'name': "Mindful resource usage"
            },
            {
                'func': self._measure_truthful_design,
                'weight': 0.2,
                'name': "Truthful communication in code"
            }
        ]
    
    def set_weights(self, **kwargs):
        """
        Set weights for different fitness objectives.
        
        Args:
            **kwargs: Weight values keyed by objective name
        """
        for key, value in kwargs.items():
            if key in self.weights:
                self.weights[key] = value
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def add_test_case(self, input_data, expected_output, weight=1.0, name=None):
        """
        Add a test case for functional evaluation.
        
        Args:
            input_data: Input data for the test case
            expected_output: Expected output for the test case
            weight: Relative importance of this test case
            name: Optional name for the test case
        """
        test_case = {
            'input': input_data,
            'expected': expected_output,
            'weight': weight,
            'name': name or f"Test case {len(self.test_cases) + 1}"
        }
        self.test_cases.append(test_case)
    
    def add_resource_constraint(self, **kwargs):
        """
        Add resource constraints for efficiency evaluation.
        
        Args:
            **kwargs: Constraint values keyed by constraint name
        """
        for key, value in kwargs.items():
            if key in self.resource_constraints:
                self.resource_constraints[key] = value
    
    def add_alignment_measure(self, measure_func, weight=1.0, name=None):
        """
        Add a measure for alignment evaluation.
        
        Args:
            measure_func: Function that takes a genome and returns an alignment score (0-1)
            weight: Relative importance of this measure
            name: Optional name for the measure
        """
        measure = {
            'func': measure_func,
            'weight': weight,
            'name': name or f"Alignment measure {len(self.alignment_measures) + 1}"
        }
        self.alignment_measures.append(measure)
    
    def add_ethical_boundary(self, boundary_name, **params):
        """
        Add an ethical boundary to the evaluator.
        
        Args:
            boundary_name: Name of the boundary to add
            **params: Additional parameters for the boundary
        """
        if self.ethical_filter:
            self.ethical_filter.add_boundary(boundary_name, **params)
        else:
            logger.warning("No ethical filter set. Cannot add boundary.")
    
    def check_ethical_boundaries(self, genome) -> bool:
        """
        Check if a genome passes all ethical boundaries.
        
        Args:
            genome: The genome to check
            
        Returns:
            True if all boundaries pass, False otherwise
        """
        if self.ethical_filter:
            return self.ethical_filter.check(genome)
        return True
    
    def evaluate(self, genome) -> float:
        """
        Evaluate a genome based on multiple objectives.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Combined fitness score
        """
        # Check ethical boundaries if filter is available
        if self.ethical_filter and not self.ethical_filter.check(genome):
            return float('-inf')  # Automatic disqualification
        
        # Evaluate each objective
        functionality_score = self._evaluate_functionality(genome)
        alignment_score = self._evaluate_alignment(genome)
        efficiency_score = self._evaluate_efficiency(genome)
        
        # Combine scores using weights
        combined_score = (
            self.weights['functionality'] * functionality_score +
            self.weights['alignment'] * alignment_score +
            self.weights['efficiency'] * efficiency_score
        )
        
        # Log detailed evaluation if in debug mode
        logger.debug(f"Fitness evaluation: functionality={functionality_score:.4f}, "
                    f"alignment={alignment_score:.4f}, efficiency={efficiency_score:.4f}, "
                    f"combined={combined_score:.4f}")
        
        return combined_score
    
    def _evaluate_functionality(self, genome) -> float:
        """
        Evaluate functional correctness using test cases.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Functionality score (0-1)
        """
        if not self.test_cases:
            return 0.5  # Neutral score if no test cases are defined
        
        # Get executable function from genome
        try:
            exec_func = self._get_executable_function(genome)
            if exec_func is None:
                return 0.0
        except Exception as e:
            logger.debug(f"Error getting executable function: {str(e)}")
            return 0.0
        
        # Run test cases
        total_weight = sum(tc['weight'] for tc in self.test_cases)
        weighted_score = 0.0
        
        for tc in self.test_cases:
            try:
                # Run the test case
                if isinstance(tc['input'], dict):
                    actual_output = exec_func(**tc['input'])
                elif isinstance(tc['input'], (list, tuple)):
                    actual_output = exec_func(*tc['input'])
                else:
                    actual_output = exec_func(tc['input'])
                
                # Check the result
                test_passed = self._compare_outputs(actual_output, tc['expected'])
                test_score = 1.0 if test_passed else 0.0
                weighted_score += test_score * tc['weight']
                
            except Exception as e:
                logger.debug(f"Error running test case {tc['name']}: {str(e)}")
                # Test case failed due to exception
                weighted_score += 0.0
        
        # Normalize score
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_alignment(self, genome) -> float:
        """
        Evaluate alignment with principles.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Alignment score (0-1)
        """
        if not self.alignment_measures:
            return 0.5  # Neutral score if no alignment measures are defined
        
        # Run alignment measures
        total_weight = sum(measure['weight'] for measure in self.alignment_measures)
        weighted_score = 0.0
        
        for measure in self.alignment_measures:
            try:
                # Run the alignment measure
                measure_score = measure['func'](genome)
                
                # Ensure score is in [0, 1] range
                measure_score = max(0.0, min(1.0, measure_score))
                weighted_score += measure_score * measure['weight']
                
            except Exception as e:
                logger.debug(f"Error running alignment measure {measure['name']}: {str(e)}")
                # Measure failed due to exception
                weighted_score += 0.0
        
        # Normalize score
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_efficiency(self, genome) -> float:
        """
        Evaluate resource efficiency.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Efficiency score (0-1)
        """
        # Get executable function from genome
        try:
            exec_func = self._get_executable_function(genome)
            if exec_func is None:
                return 0.0
        except Exception as e:
            logger.debug(f"Error getting executable function: {str(e)}")
            return 0.0
        
        # Measure execution time
        time_score = self._measure_execution_time(exec_func)
        
        # Measure memory usage
        memory_score = self._measure_memory_usage(exec_func)
        
        # Combine metrics
        return 0.5 * (time_score + memory_score)
    
    def _measure_execution_time(self, func) -> float:
        """
        Measure execution time and compute a normalized score.
        
        Args:
            func: Function to measure
            
        Returns:
            Time efficiency score (0-1)
        """
        if not self.test_cases:
            # No test cases to run
            return 0.5
        
        max_time = self.resource_constraints.get('max_execution_time', 1.0)
        total_time = 0.0
        num_cases = 0
        
        for tc in self.test_cases:
            try:
                # Measure execution time
                start_time = time.time()
                
                if isinstance(tc['input'], dict):
                    func(**tc['input'])
                elif isinstance(tc['input'], (list, tuple)):
                    func(*tc['input'])
                else:
                    func(tc['input'])
                
                execution_time = time.time() - start_time
                total_time += execution_time
                num_cases += 1
                
            except Exception:
                # Ignore errors in measuring time
                pass
        
        if num_cases == 0:
            return 0.5
        
        # Calculate average time
        avg_time = total_time / num_cases
        
        # Normalize score (lower is better)
        # 1.0 if time is 0, 0.0 if time is max_time or above
        return max(0.0, 1.0 - (avg_time / max_time))
    
    def _measure_memory_usage(self, func) -> float:
        """
        Measure memory usage and compute a normalized score.
        
        Args:
            func: Function to measure
            
        Returns:
            Memory efficiency score (0-1)
        """
        if not self.test_cases:
            # No test cases to run
            return 0.5
        
        max_memory = self.resource_constraints.get('max_memory_usage', 100.0)  # MB
        peak_memory = 0.0
        
        for tc in self.test_cases[:1]:  # Just use the first test case for memory measurement
            try:
                # Start memory tracking
                tracemalloc.start()
                
                if isinstance(tc['input'], dict):
                    func(**tc['input'])
                elif isinstance(tc['input'], (list, tuple)):
                    func(*tc['input'])
                else:
                    func(tc['input'])
                
                # Get peak memory
                _, peak = tracemalloc.get_traced_memory()
                peak_memory = peak / (1024 * 1024)  # Convert to MB
                
                # Stop tracking
                tracemalloc.stop()
                
            except Exception:
                # Stop tracking in case of exception
                tracemalloc.stop()
        
        # Normalize score (lower is better)
        # 1.0 if memory is 0, 0.0 if memory is max_memory or above
        return max(0.0, 1.0 - (peak_memory / max_memory))
    
    def _get_executable_function(self, genome) -> Optional[Callable]:
        """
        Get an executable function from a genome.
        
        Args:
            genome: The genome to execute
            
        Returns:
            Callable function or None if execution fails
        """
        # Get source code from genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        # Create a local execution environment
        local_env = {}
        
        try:
            # Execute the code in the local environment
            exec(source_code, {}, local_env)
            
            # Find the first function defined in the code
            for name, obj in local_env.items():
                if inspect.isfunction(obj):
                    return obj
            
            # No functions found
            logger.debug("No functions found in genome")
            return None
            
        except Exception as e:
            logger.debug(f"Error executing genome: {str(e)}")
            return None
    
    def _compare_outputs(self, actual, expected) -> bool:
        """
        Compare actual and expected outputs with some flexibility.
        
        Args:
            actual: Actual output from the function
            expected: Expected output from the test case
            
        Returns:
            True if outputs match, False otherwise
        """
        # Direct equality
        if actual == expected:
            return True
        
        # Type conversion for numeric values
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # Allow small differences for floating point values
            if abs(float(actual) - float(expected)) < 1e-6:
                return True
        
        # List/tuple comparison with some flexibility
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(expected) == len(actual):
                return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        # Dictionary comparison
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected.keys()) == set(actual.keys()):
                return all(self._compare_outputs(actual[k], expected[k]) for k in expected)
        
        return False
    
    # New alignment measure methods
    def _measure_code_clarity(self, genome) -> float:
        """
        Measure code clarity, simplicity and readability.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Score from 0-1 where 1 is most clear
        """
        # Get source code from genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        try:
            # Parse the AST
            tree = ast.parse(source_code)
        except SyntaxError:
            return 0.0  # Invalid code gets lowest score
        
        # Start with a perfect score and deduct for issues
        score = 1.0
        
        # 1. Check comment ratio (aim for 10-30% comments)
        comment_lines = 0
        code_lines = 0
        for line in source_code.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_lines += 1
            elif stripped:
                code_lines += 1
                if '#' in stripped:
                    comment_lines += 1
        
        total_lines = code_lines + comment_lines
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            # Penalize if comments are too sparse or too dense
            if comment_ratio < 0.1:
                score -= 0.1
            elif comment_ratio > 0.4:
                score -= 0.05
        
        # 2. Function length (prefer functions under 30 lines)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_source = ast.unparse(node)
                function_lines = function_source.count('\n') + 1
                if function_lines > 30:
                    score -= min(0.2, (function_lines - 30) / 100)
        
        # 3. Variable name quality
        variable_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                name = node.id
                variable_lengths.append(len(name))
                # Penalize single-letter variables except for common loop indices
                if len(name) == 1 and name not in ['i', 'j', 'k', 'x', 'y', 'n']:
                    score -= 0.05
                # Penalize excessively long variable names
                elif len(name) > 30:
                    score -= 0.05
        
        # Calculate average variable name length
        if variable_lengths:
            avg_length = sum(variable_lengths) / len(variable_lengths)
            # Ideal average length is around 8-15 characters
            if avg_length < 5:
                score -= 0.1
        
        # 4. Complexity measures
        complexity = 0
        nesting_depth = 0
        for node in ast.walk(tree):
            # Count control structures for complexity
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            
            # Calculate maximum nesting depth
            if isinstance(node, (ast.If, ast.For, ast.While)):
                depth = self._calculate_nesting_depth(node)
                nesting_depth = max(nesting_depth, depth)
        
        # Penalize excessive complexity
        if complexity > 20:
            score -= min(0.2, (complexity - 20) / 100)
        
        # Penalize deep nesting
        if nesting_depth > 3:
            score -= min(0.2, (nesting_depth - 3) / 10)
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    
    def _calculate_nesting_depth(self, node, current_depth=1):
        """Helper function to calculate nesting depth"""
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _measure_inclusive_language(self, genome) -> float:
        """
        Measure how inclusive and respectful the code language is.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Score from 0-1 where 1 is most inclusive
        """
        # Get source code from genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        # Start with a perfect score and deduct for issues
        score = 1.0
        
        # Define problematic terms
        problematic_terms = [
            # Binary thinking terms
            'master', 'slave', 'blacklist', 'whitelist',
            # Potentially exclusionary terms
            'guys', 'manpower', 'mankind',
            # Hierarchical terms
            'superior', 'inferior', 'subordinate'
        ]
        
        # Define preferred alternatives (for educative purposes)
        alternatives = {
            'master': 'main, primary, leader',
            'slave': 'secondary, replica, follower',
            'blacklist': 'blocklist, denylist',
            'whitelist': 'allowlist, safelist',
            'guys': 'folks, team, everyone',
            'manpower': 'workforce, staff, personnel',
            'mankind': 'humanity, people',
            'superior': 'preceding, previous',
            'inferior': 'following, subsequent'
        }
        
        # Check for problematic terms in code, comments, strings, and identifiers
        for term in problematic_terms:
            pattern = r'\b' + term + r'\b'
            matches = re.finditer(pattern, source_code, re.IGNORECASE)
            if any(matches):
                score -= 0.1  # Deduct for each type of problematic term
        
        # Check if code promotes inclusivity (positive points)
        positive_indicators = [
            'accessible', 'inclusive', 'diversity', 'respect', 'inclusion'
        ]
        
        positive_score = 0
        for term in positive_indicators:
            pattern = r'\b' + term + r'\b'
            matches = list(re.finditer(pattern, source_code, re.IGNORECASE))
            positive_score += min(0.05 * len(matches), 0.1)  # Cap at 0.1 per term
        
        # Add positive indicators to score (up to 0.2 total)
        score += min(positive_score, 0.2)
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    
    def _measure_service_orientation(self, genome) -> float:
        """
        Measure how well the code serves others rather than self-serving.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Score from 0-1 where 1 is most service-oriented
        """
        # Get source code from genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        try:
            # Parse the AST
            tree = ast.parse(source_code)
        except SyntaxError:
            return 0.0  # Invalid code gets lowest score
        
        # Start with a neutral score
        score = 0.5
        
        # 1. Check docstring completeness and helpfulness
        docstring_score = 0
        total_functions = 0
        documented_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                docstring = ast.get_docstring(node)
                if docstring:
                    documented_functions += 1
                    # Check docstring quality
                    if "Args:" in docstring or "Parameters:" in docstring:
                        docstring_score += 0.5
                    if "Returns:" in docstring:
                        docstring_score += 0.5
                    # Reward longer, more helpful docstrings
                    lines = docstring.strip().count('\n') + 1
                    if lines >= 3:
                        docstring_score += min(lines / 10, 0.5)
        
        # Calculate average docstring score
        if total_functions > 0:
            avg_docstring_score = docstring_score / total_functions
            documentation_ratio = documented_functions / total_functions
            score += 0.2 * avg_docstring_score
            score += 0.1 * documentation_ratio
        
        # 2. Error handling and user-friendly messages
        error_handling_score = 0
        try_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_count += 1
                
                # Check for helpful error messages in except blocks
                for handler in node.handlers:
                    for subnode in ast.walk(handler):
                        if isinstance(subnode, (ast.Raise, ast.Return)) and hasattr(subnode, 'value'):
                            if isinstance(subnode.value, ast.Call) and hasattr(subnode.value.func, 'id'):
                                if subnode.value.func.id in ['Exception', 'ValueError', 'RuntimeError']:
                                    # Check if custom error message is provided
                                    if subnode.value.args and isinstance(subnode.value.args[0], ast.Str):
                                        error_handling_score += 0.5
        
        # Add error handling score (max 0.1)
        if try_count > 0:
            score += min(0.1, error_handling_score / (try_count * 2))
        
        # 3. Check for input validation
        validation_score = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body_code = ast.unparse(node)
                # Look for validation patterns like type checks or assertions
                validation_patterns = [
                    r'isinstance\(', r'assert\s+', r'if\s+.+\s+is\s+(not\s+)?None',
                    r'if\s+not\s+.+:', r'if\s+len\(.+\)\s*[<>=]'
                ]
                
                for pattern in validation_patterns:
                    if re.search(pattern, body_code):
                        validation_score += 0.5
                        break
        
        # Add validation score (max 0.2)
        if total_functions > 0:
            score += min(0.2, validation_score / total_functions)
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    
    def _measure_mindful_resource_usage(self, genome) -> float:
        """
        Measure how mindfully the code uses computational resources.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Score from 0-1 where 1 is most mindful
        """
        # Get source code from genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        try:
            # Parse the AST
            tree = ast.parse(source_code)
        except SyntaxError:
            return 0.0  # Invalid code gets lowest score
        
        # Start with a perfect score and deduct for issues
        score = 1.0
        
        # 1. Check for inefficient patterns
        inefficient_patterns = [
            # String concatenation in loops
            (r'for\s+.+:\s*\n\s+.+\s*\+=\s*[\'"]\w*[\'"]', 
             "String concatenation in loop (use join instead)"),
            
            # Multiple list/dict comprehensions that could be combined
            (r'\[[^\]]+\]\s*\n\s*\[[^\]]+\]', 
             "Multiple list comprehensions could be combined"),
            
            # Nested loops with O(n²) or worse complexity
            (r'for\s+.+:\s*\n\s+for\s+.+:\s*\n\s+for\s+.+:', 
             "Triple nested loop (O(n³) complexity)"),
            
            # Creating large temporary collections
            (r'range\(\d{5,}\)', 
             "Very large range created"),
            
            # Repeatedly calling expensive functions
            (r'for\s+.+:\s*\n\s+.+\(.*sorted\(', 
             "Sorting inside a loop")
        ]
        
        # Check inefficient patterns
        for pattern, message in inefficient_patterns:
            if re.search(pattern, source_code, re.MULTILINE):
                score -= 0.1
        
        # 2. Check for resource management
        file_handles = []
        file_closings = []
        
        for node in ast.walk(tree):
            # Check for file opens
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
                file_handles.append(node)
            
            # Check for with statements (good practice)
            if isinstance(node, ast.With):
                for item in node.items:
                    if (isinstance(item.context_expr, ast.Call) and 
                        isinstance(item.context_expr.func, ast.Name) and 
                        item.context_expr.func.id == 'open'):
                        file_closings.append(item)
            
            # Check for explicit file closes
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and 
                node.func.attr == 'close'):
                file_closings.append(node)
        
        # Penalize unclosed file handles
        if len(file_handles) > len(file_closings):
            score -= 0.2
        
        # 3. Check for memory efficiency
        memory_wasteful_patterns = 0
        
        for node in ast.walk(tree):
            # Unnecessarily large data structures
            if isinstance(node, ast.Dict) and len(node.keys) > 100:
                memory_wasteful_patterns += 1
            
            # Large literal collections
            if isinstance(node, ast.List) and len(node.elts) > 100:
                memory_wasteful_patterns += 1
        
        # Penalize memory wastefulness
        score -= min(0.3, memory_wasteful_patterns * 0.1)
        
        # 4. Check for CPU-friendly code
        cpu_intensive_operations = 0
        
        for node in ast.walk(tree):
            # Recursive calls without memoization
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                # Check if function calls itself
                for subnode in ast.walk(node):
                    if (isinstance(subnode, ast.Call) and 
                        isinstance(subnode.func, ast.Name) and 
                        subnode.func.id == function_name):
                        # Check if there's no memoization
                        function_body = ast.unparse(node)
                        if 'cache' not in function_body and 'memo' not in function_body:
                            cpu_intensive_operations += 1
        
        # Penalize CPU-intensive operations
        score -= min(0.2, cpu_intensive_operations * 0.1)
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    
    def _measure_truthful_design(self, genome) -> float:
        """
        Measure how truthful and transparent the code is in its communication.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Score from 0-1 where 1 is most truthful
        """
        # Get source code from genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        try:
            # Parse the AST
            tree = ast.parse(source_code)
        except SyntaxError:
            return 0.0  # Invalid code gets lowest score
        
        # Start with a perfect score and deduct for issues
        score = 1.0
        
        # 1. Check for misleading function names
        misleading_name_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name.lower()
                docstring = ast.get_docstring(node)
                function_body = ast.unparse(node.body)
                
                # Check if name suggests one action but does another
                if ('get' in name) and ('set' in function_body or 'save' in function_body):
                    misleading_name_count += 1
                elif ('is' in name or 'has' in name) and 'return ' not in function_body:
                    misleading_name_count += 1
                elif ('validate' in name) and 'return True' in function_body and 'if' not in function_body:
                    misleading_name_count += 1
        
        # Penalize misleading names
        score -= min(0.3, misleading_name_count * 0.1)
        
        # 2. Check for actual/claimed behavior consistency
        behavior_mismatch_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Check for docstring claims that don't match implementation
                    if 'fast' in docstring.lower() or 'efficient' in docstring.lower():
                        function_body = ast.unparse(node.body)
                        if 'for' in function_body and 'for' in function_body[function_body.find('for')+3:]:
                            # Nested loops in a function claiming to be fast
                            behavior_mismatch_count += 1
                    
                    # Check for docstring promises vs. exceptions
                    if 'always' in docstring.lower() or 'guaranteed' in docstring.lower():
                        function_body = ast.unparse(node.body)
                        if 'raise' in function_body:
                            # Function claims to always work but can raise exceptions
                            behavior_mismatch_count += 1
        
        # Penalize behavior mismatches
        score -= min(0.3, behavior_mismatch_count * 0.1)
        
        # 3. Check for transparent error handling
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    # Check for empty except blocks or generic exceptions
                    if not handler.body or len(handler.body) <= 1:
                        score -= 0.1
                    if handler.type is None or (isinstance(handler.type, ast.Name) and handler.type.id == 'Exception'):
                        score -= 0.05
                    # Check for exception suppression
                        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                            score -= 0.1
        
        # 4. Check for honest naming patterns
        variable_count = 0
        descriptive_names = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variable_count += 1
                name = node.id
                # Check if name is descriptive enough
                if len(name) > 3 and not name.startswith('_'):
                    descriptive_names += 1
        
        # Reward descriptive naming
        if variable_count > 0:
            descriptive_ratio = descriptive_names / variable_count
            score += 0.1 * descriptive_ratio
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    
    def get_detailed_evaluation(self, genome) -> Dict[str, Any]:
        """
        Get detailed evaluation metrics for a genome.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Dictionary with detailed evaluation metrics
        """
        # Check ethical boundaries if filter is available
        ethical_pass = True
        ethical_violations = {}
        if self.ethical_filter:
            ethical_pass = self.ethical_filter.check(genome)
            if not ethical_pass:
                ethical_violations = self.ethical_filter.explain_violations(genome)
        
        # If ethical boundaries fail, return early with violations
        if not ethical_pass:
            return {
                'passed_ethical_boundaries': False,
                'ethical_violations': ethical_violations,
                'fitness': float('-inf')
            }
        
        # Evaluate each objective
        functionality_score = self._evaluate_functionality(genome)
        alignment_score = self._evaluate_alignment(genome)
        efficiency_score = self._evaluate_efficiency(genome)
        
        # Combine scores using weights
        combined_score = (
            self.weights['functionality'] * functionality_score +
            self.weights['alignment'] * alignment_score +
            self.weights['efficiency'] * efficiency_score
        )
        
        # Get detailed test case results
        test_results = []
        exec_func = self._get_executable_function(genome)
        
        if exec_func:
            for tc in self.test_cases:
                try:
                    # Run the test case
                    if isinstance(tc['input'], dict):
                        actual_output = exec_func(**tc['input'])
                    elif isinstance(tc['input'], (list, tuple)):
                        actual_output = exec_func(*tc['input'])
                    else:
                        actual_output = exec_func(tc['input'])
                    
                    # Check the result
                    passed = self._compare_outputs(actual_output, tc['expected'])
                    
                    test_results.append({
                        'name': tc['name'],
                        'passed': passed,
                        'input': tc['input'],
                        'expected': tc['expected'],
                        'actual': actual_output
                    })
                    
                except Exception as e:
                    test_results.append({
                        'name': tc['name'],
                        'passed': False,
                        'input': tc['input'],
                        'expected': tc['expected'],
                        'error': str(e)
                    })
        
        # Return detailed evaluation
        return {
            'passed_ethical_boundaries': True,
            'fitness': combined_score,
            'objective_scores': {
                'functionality': functionality_score,
                'alignment': alignment_score,
                'efficiency': efficiency_score
            },
            'test_results': test_results,
            'objective_weights': self.weights
        } 