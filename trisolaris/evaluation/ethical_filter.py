"""
Ethical Boundary Enforcer for the TRISOLARIS framework.

This module implements strict ethical filters that enforce boundaries on what
evolved code can do, ensuring it operates within safe and ethical constraints.
"""

import ast
import re
import logging
import inspect
from typing import List, Dict, Set, Optional, Any, Callable, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EthicalBoundaryEnforcer:
    """
    Enforces ethical boundaries on evolved code to ensure safety and ethical compliance.
    
    This class implements a set of filters that analyze code for potential safety
    issues and ethical concerns before it is evaluated or executed.
    """
    
    def __init__(self):
        """Initialize the ethical boundary enforcer."""
        self.boundaries = {}
        self.boundary_results = {}
        self.violation_history = []
        
        # Define standard boundary checks
        self._standard_boundaries = {
            # Safety-critical boundaries
            "no_system_calls": self._check_no_system_calls,
            "no_eval_exec": self._check_no_eval_exec,
            "no_file_operations": self._check_no_file_operations,
            "no_network_access": self._check_no_network_access,
            "no_imports": self._check_no_imports,
            "max_execution_time": self._check_max_execution_time,
            "max_memory_usage": self._check_max_memory_usage,
            
            # Gurbani-inspired ethical boundaries
            "universal_equity": self._check_universal_equity,
            "truthful_communication": self._check_truthful_communication,
            "humble_code": self._check_humble_code,
            "service_oriented": self._check_service_oriented,
            "harmony_with_environment": self._check_harmony_with_environment
        }
    
    def check(self, genome) -> bool:
        """
        Check if a genome passes all ethical boundaries. This is an alias for check_all_boundaries.
        
        Args:
            genome: The genome to check
            
        Returns:
            True if all boundaries pass, False otherwise
        """
        return self.check_all_boundaries(genome)
    
    def add_boundary(self, boundary_name: str, **kwargs) -> None:
        """
        Add an ethical boundary check to the enforcer.
        
        Args:
            boundary_name: Name of the boundary to add
            **kwargs: Parameters for the boundary check
        """
        if boundary_name in self._standard_boundaries:
            self.boundaries[boundary_name] = {
                'check': self._standard_boundaries[boundary_name],
                'params': kwargs
            }
            logger.info(f"Added boundary: {boundary_name}")
        else:
            logger.error(f"Unknown boundary: {boundary_name}")
    
    def add_custom_boundary(self, name: str, check_function: Callable, **kwargs) -> None:
        """
        Add a custom boundary check.
        
        Args:
            name: Name for the custom boundary
            check_function: Function that implements the check
            **kwargs: Parameters for the check function
        """
        self.boundaries[name] = {
            'check': check_function,
            'params': kwargs
        }
        logger.info(f"Added custom boundary: {name}")
    
    def check_all_boundaries(self, genome) -> bool:
        """
        Check if a genome satisfies all defined ethical boundaries.
        
        Args:
            genome: The genome to check
            
        Returns:
            True if all boundaries are satisfied, False otherwise
        """
        self.boundary_results = {}
        all_passed = True
        
        # Get the source code from the genome
        try:
            source = genome.to_source()
        except Exception as e:
            logger.error(f"Failed to get source from genome: {str(e)}")
            return False
        
        # Apply each boundary check
        for name, boundary in self.boundaries.items():
            try:
                check_func = boundary['check']
                params = boundary['params']
                passed = check_func(source, **params)
                self.boundary_results[name] = passed
                
                if not passed:
                    logger.warning(f"Boundary violation: {name}")
                    self.violation_history.append({
                        'boundary': name,
                        'source_excerpt': source[:200] + ('...' if len(source) > 200 else '')
                    })
                    all_passed = False
            except Exception as e:
                logger.error(f"Error checking boundary '{name}': {str(e)}")
                self.boundary_results[name] = False
                all_passed = False
        
        return all_passed
    
    def get_boundary_results(self) -> Dict[str, bool]:
        """
        Get the results of the last boundary check.
        
        Returns:
            Dictionary mapping boundary names to check results (True/False)
        """
        return self.boundary_results
    
    def get_active_boundaries(self) -> List[str]:
        """
        Get the names of all active boundaries.
        
        Returns:
            List of active boundary names
        """
        return list(self.boundaries.keys())
    
    def get_violation_history(self) -> List[Dict]:
        """
        Get the history of boundary violations.
        
        Returns:
            List of violation records
        """
        return self.violation_history
    
    def clear_violation_history(self) -> None:
        """Clear the violation history."""
        self.violation_history = []
    
    # === Safety-Critical Boundary Checks ===
    
    def _check_no_system_calls(self, source: str, **kwargs) -> bool:
        """Check that the code doesn't attempt to make system calls."""
        # Check for os.system, subprocess, etc.
        patterns = [
            r'\bos\.system\s*\(',
            r'\bsubprocess\.',
            r'\bshell\s*=\s*True',
            r'\bcommand\s*\(',
            r'\bpopen\s*\(',
            r'`[^`]*`',  # Backticks for command substitution
            r'\bexec\s*\(',
            r'\bshutil\..*exec',
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return False
        
        return True
    
    def _check_no_eval_exec(self, source: str, **kwargs) -> bool:
        """Check that the code doesn't use eval() or exec()."""
        patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bcompile\s*\([^)]*\bexec\b',
            r'__import__\s*\(',
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return False
        
        return True
    
    def _check_no_file_operations(self, source: str, allowed_dirs: List[str] = None, **kwargs) -> bool:
        """
        Check that the code doesn't perform file operations outside allowed directories.
        
        Args:
            source: Code source to check
            allowed_dirs: List of directories where file operations are allowed
        """
        if allowed_dirs is None:
            # If no directories are explicitly allowed, check for any file operations
            patterns = [
                r'\bopen\s*\(',
                r'\.write\s*\(',
                r'\.read\s*\(',
                r'\.writelines\s*\(',
                r'\.readlines\s*\(',
                r'\bos\.remove\s*\(',
                r'\bos\.unlink\s*\(',
                r'\bos\.rename\s*\(',
                r'\bos\.makedirs\s*\(',
                r'\bshutil\.copy',
                r'\bshutil\.move',
                r'\bpathlib\.Path'
            ]
            
            for pattern in patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    return False
            
            return True
        else:
            # TODO: Implement more sophisticated check for allowed directories
            # This would require parsing the AST to extract file paths and check them
            return True
    
    def _check_no_network_access(self, source: str, **kwargs) -> bool:
        """Check that the code doesn't attempt network access."""
        patterns = [
            r'\bsocket\.',
            r'\burllib\.',
            r'\brequests\.',
            r'\bhttp\.',
            r'\burllib2\.',
            r'\bftplib\.',
            r'\.connect\s*\(',
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return False
        
        return True
    
    def _check_no_imports(self, source: str, allowed_imports: Set[str] = None, **kwargs) -> bool:
        """
        Check that the code only imports from an allowed list of modules.
        
        Args:
            source: Code source to check
            allowed_imports: Set of allowed import module names
        """
        if allowed_imports is None:
            allowed_imports = set()
        
        try:
            tree = ast.parse(source)
            
            # Check all import statements
            for node in ast.walk(tree):
                # Regular imports: import x, import x.y
                if isinstance(node, ast.Import):
                    for name in node.names:
                        root_module = name.name.split('.')[0]
                        if root_module not in allowed_imports:
                            logger.warning(f"Disallowed import: {root_module}")
                            return False
                
                # From imports: from x import y, from x.y import z
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_module = node.module.split('.')[0]
                        if root_module not in allowed_imports:
                            logger.warning(f"Disallowed import from: {root_module}")
                            return False
            
            return True
        except SyntaxError:
            # If we can't parse the code, fail closed for safety
            logger.warning("Syntax error while checking imports")
            return False
    
    def _check_max_execution_time(self, source: str, max_execution_time: float = 1.0, **kwargs) -> bool:
        """
        Check if code contains safety measures for execution time limits.
        
        This is a static check that looks for patterns indicating time-consuming operations.
        The actual runtime enforcement needs to happen during execution.
        
        Args:
            source: Code source to check
            max_execution_time: Maximum allowed execution time in seconds
        """
        # Look for potential infinite loops, heavy recursion, etc.
        patterns = [
            r'while\s+True\s*:',
            r'for\s+[^:]+\s+in\s+range\s*\(\s*[0-9]{7,}\s*\)',  # Very large ranges
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                # Check if there's a timeout mechanism
                if not re.search(r'time\.(sleep|time)\s*\(', source, re.IGNORECASE):
                    logger.warning(f"Potential infinite loop without timeout")
                    return False
        
        return True
    
    def _check_max_memory_usage(self, source: str, max_memory_usage: int = 100, **kwargs) -> bool:
        """
        Check if code contains patterns that might lead to excessive memory usage.
        
        This is a static check that looks for patterns indicating memory-intensive operations.
        The actual memory enforcement needs to happen during execution.
        
        Args:
            source: Code source to check
            max_memory_usage: Maximum allowed memory usage in MB
        """
        # Look for large data structures, etc.
        patterns = [
            r'range\s*\(\s*[0-9]{7,}\s*\)',  # Very large ranges
            r'\[\s*[0-9]+\s*\*\s*[0-9]{6,}\s*\]',  # Large list comprehensions
            r'\{\s*[0-9]+\s*\:\s*[0-9]+\s*for\s+[^}]+\s+in\s+range\s*\(\s*[0-9]{6,}\s*\)\s*\}',  # Large dict comprehensions
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"Potential excessive memory usage")
                return False
        
        return True
    
    # === Gurbani-Inspired Ethical Boundaries ===
    
    def _check_universal_equity(self, source: str, **kwargs) -> bool:
        """
        Check for code patterns that might treat different inputs inequitably.
        
        Inspired by the Gurbani principle of treating all equally.
        """
        # Look for hardcoded biases, special cases for specific groups, etc.
        patterns = [
            r'\bif\s+.*\b(gender|race|ethnicity|nationality|religion)\b.*:',  # Simplistic check for demographic-based branching
            r'bias',
            r'blacklist',
            r'whitelist',
        ]
        
        # This is a simplified check - a real implementation would need more sophisticated analysis
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                # Look for comments indicating ethical considerations
                context = 10  # Lines of context to check
                matches = re.finditer(pattern, source, re.IGNORECASE)
                for match in matches:
                    start_pos = match.start()
                    # Find the start of the line containing this match
                    line_start = source.rfind('\n', 0, start_pos) + 1
                    # Find several lines before
                    previous_lines = source[max(0, source.rfind('\n', 0, line_start - context)):line_start]
                    
                    # If there's a comment indicating ethical consideration, allow it
                    if re.search(r'#.*\b(ethical|fairness|equity|equality)\b', previous_lines, re.IGNORECASE):
                        continue
                        
                    logger.warning(f"Potential equity issue found: {match.group(0)}")
                    return False
        
        return True
    
    def _check_truthful_communication(self, source: str, **kwargs) -> bool:
        """
        Check for code patterns that might lead to misleading outputs.
        
        Inspired by the Gurbani principle of truthfulness (Sat Nam).
        """
        # Look for deceptive patterns, misleading variable names, etc.
        patterns = [
            r'fake',
            r'deceive',
            r'trick',
            r'mislead',
            r'false.*report',
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"Potential truthfulness issue found: {pattern}")
                return False
        
        # Check for error handling - code should handle and report errors truthfully
        try:
            tree = ast.parse(source)
            has_try = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    has_try = True
                    break
            
            # If code has function definitions but no error handling, flag it
            # This is a simplified heuristic
            has_funcs = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            if has_funcs and not has_try and len(source.splitlines()) > 30:
                logger.warning("Code lacks error handling for truthful communication")
                return False
                
        except SyntaxError:
            # If we can't parse the code, pass this check but log a warning
            logger.warning("Syntax error while checking for truthful communication")
        
        return True
    
    def _check_humble_code(self, source: str, **kwargs) -> bool:
        """
        Check for code patterns that might indicate unnecessary complexity or "showing off."
        
        Inspired by the Gurbani principle of humility (Nimrata).
        """
        # Look for unnecessarily complex code, "clever tricks," etc.
        # Calculate code complexity metrics
        
        # Check line length - consistently very long lines might indicate lack of readability
        lines = source.splitlines()
        very_long_lines = sum(1 for line in lines if len(line.strip()) > 100)
        if lines and very_long_lines / len(lines) > 0.3:  # More than 30% are very long
            logger.warning("Code has many excessively long lines, reducing readability")
            return False
        
        # Check for nested list/dict/set comprehensions (often less readable)
        if re.search(r'\[.*\bfor\b.*\bfor\b.*\]', source) or re.search(r'\{.*\bfor\b.*\bfor\b.*\}', source):
            # Check if there are comments explaining the complex comprehensions
            if not re.search(r'#.*comprehension', source, re.IGNORECASE):
                logger.warning("Complex comprehensions without explanatory comments")
                return False
        
        # Check for deep nesting of control structures
        try:
            tree = ast.parse(source)
            
            class NestingVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.max_nesting = 0
                    self.current_nesting = 0
                
                def generic_visit(self, node):
                    if isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                        self.current_nesting += 1
                        if self.current_nesting > self.max_nesting:
                            self.max_nesting = self.current_nesting
                        super().generic_visit(node)
                        self.current_nesting -= 1
                    else:
                        super().generic_visit(node)
            
            visitor = NestingVisitor()
            visitor.visit(tree)
            
            if visitor.max_nesting > 5:  # Arbitrary threshold
                logger.warning(f"Code has deep nesting level: {visitor.max_nesting}")
                return False
                
        except SyntaxError:
            # If we can't parse the code, pass this check but log a warning
            logger.warning("Syntax error while checking for humble code")
        
        return True
    
    def _check_service_oriented(self, source: str, **kwargs) -> bool:
        """
        Check if code appears to be service-oriented rather than self-serving.
        
        Inspired by the Gurbani principle of selfless service (Seva).
        """
        # Look for proper documentation, user-focused error messages, etc.
        
        # Check for docstrings
        try:
            tree = ast.parse(source)
            
            # Check top-level module docstring
            has_module_docstring = False
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                has_module_docstring = True
            
            # Check function docstrings
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            functions_with_docstrings = 0
            for func in functions:
                if func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Str):
                    functions_with_docstrings += 1
            
            # If there are functions but less than half have docstrings, flag it
            if functions and functions_with_docstrings / len(functions) < 0.5:
                logger.warning("Less than half of functions have docstrings")
                return False
            
            # If it's a significant module with no module docstring, flag it
            if len(tree.body) > 5 and not has_module_docstring:
                logger.warning("Significant module with no top-level docstring")
                return False
                
        except SyntaxError:
            # If we can't parse the code, pass this check but log a warning
            logger.warning("Syntax error while checking for service orientation")
        
        return True
    
    def _check_harmony_with_environment(self, source: str, **kwargs) -> bool:
        """
        Check if code is resource-efficient and environmentally conscious.
        
        Inspired by the Gurbani principle of harmony with nature.
        """
        # Look for resource-inefficient patterns, lack of cleanup, etc.
        
        # Check for resource cleanup in file operations, etc.
        patterns = [
            # File operations without context managers
            r'open\([^)]*\)[^:]*[^\n]*((?!with|close).)*$',
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.MULTILINE | re.DOTALL):
                logger.warning("Resource usage without proper cleanup detected")
                return False
        
        # Check for efficient code patterns vs. inefficient ones
        inefficient_patterns = [
            r'\.append\s*\([^)]*\)\s*in\s+range',  # Building lists with append in loops
            r'\+\=\s*"[^"]*"\s*in\s+range',  # String concatenation in loops
        ]
        
        for pattern in inefficient_patterns:
            if re.search(pattern, source):
                logger.warning(f"Inefficient code pattern detected: {pattern}")
                return False
        
        return True
    
    def __str__(self) -> str:
        """Return a string representation of the enforcer."""
        active = self.get_active_boundaries()
        return f"EthicalBoundaryEnforcer({len(active)} active boundaries: {', '.join(active)})"
