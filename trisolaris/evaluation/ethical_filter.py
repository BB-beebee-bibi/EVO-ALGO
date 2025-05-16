"""
Ethical Boundary Enforcer for the TRISOLARIS framework.

This module implements ethical filters that enforce boundaries on what
evolved code can do, ensuring it operates within safe and ethical constraints.
It provides multiple layers of protection to prevent harmful code generation
and execution, with a focus on safety, ethics, and responsible AI development.

The module now implements a post-evolution ethical evaluation approach,
which evaluates code after it has been evolved rather than constraining
the evolution process itself. This allows for more creative evolution
while still ensuring ethical compliance.
"""

import ast
import re
import logging
import inspect
import os
import time
import functools
import hashlib
from typing import List, Dict, Set, Optional, Any, Callable, Union, Tuple

# Import the new post-evolution evaluation components
from trisolaris.evaluation.syntax_checker import check_syntax, check_functionality, measure_output
from trisolaris.evaluation.ethics_client import evaluate_ethics, parse_ethics_report, check_gurbani_alignment
from trisolaris.config import get_config, BaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EthicalBoundaryEnforcer:
    """
    Enforces ethical boundaries on evolved code to ensure safety and ethical compliance.
    
    This class implements a set of filters that analyze code for potential safety
    issues and ethical concerns before it is evaluated or executed. It provides
    multiple layers of protection through both pattern-based and AST-based analysis.
    """
    
    def __init__(self, config: Optional[BaseConfig] = None, component_name: str = "ethical_boundaries", run_id: Optional[str] = None):
        """
        Initialize the ethical boundary enforcer.
        
        Args:
            config: Configuration object (if None, will be loaded from global config)
            component_name: Name of this component for configuration lookup
            run_id: Optional run ID for configuration lookup
        """
        # Load configuration
        self.config = config or get_config(component_name, run_id)
        self.component_name = component_name
        self.run_id = run_id
        
        # Initialize from configuration
        self.boundaries = {}
        self.boundary_results = {}
        self.violation_history = []
        self.use_post_evolution = self.config.ethical_boundaries.use_post_evolution
        self.evaluation_cache = {}
        
        # Define standard boundary checks (for backward compatibility)
        self._standard_boundaries = {
            # Safety-critical boundaries
            "no_system_calls": self._check_no_system_calls,
            "no_eval_exec": self._check_no_eval_exec,
            "no_file_operations": self._check_no_file_operations,
            "no_network_access": self._check_no_network_access,
            "no_imports": self._check_no_imports,
            "max_execution_time": self._check_max_execution_time,
            "max_memory_usage": self._check_max_memory_usage,
            "sandboxed_execution": self._check_sandboxed_execution
        }
        
        logger.info(f"Initialized EthicalBoundaryEnforcer with post-evolution mode: {self.use_post_evolution}")
    
    def check(self, genome) -> bool:
        """
        Check if a genome passes all ethical boundaries.
        
        In post-evolution mode, this performs a comprehensive ethical evaluation
        of the code after evolution. In pre-evolution mode (legacy), this is an
        alias for check_all_boundaries.
        
        Args:
            genome: The genome to check
            
        Returns:
            True if all ethical checks pass, False otherwise
        """
        if self.use_post_evolution:
            return self.evaluate_post_evolution(genome)
        else:
            return self.check_all_boundaries(genome)
    
    def evaluate_post_evolution(self, genome) -> bool:
        """
        Perform post-evolution ethical evaluation on a genome.
        
        This method evaluates the code after evolution using syntax checking,
        functionality verification, and ethical evaluation via the ethics client.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            True if the genome passes all post-evolution checks, False otherwise
        """
        try:
            source = genome.to_source()
            
            # Check for cached evaluation result
            cache_key = self._get_cache_key(source)
            if cache_key in self.evaluation_cache:
                logger.info(f"Using cached ethical evaluation result for genome")
                return self.evaluation_cache[cache_key]
            
            # Step 1: Check syntax
            syntax_valid, syntax_error = check_syntax(source)
            if not syntax_valid:
                logger.warning(f"Genome failed syntax check: {syntax_error}")
                self._record_violation("syntax_error", source, syntax_error)
                self.evaluation_cache[cache_key] = False
                return False
            
            # Step 2: Check functionality
            # Create a mock task for now - in a real implementation, this would be passed in
            mock_task = {"type": "generic", "expected_outputs": {}}
            functionality_valid, functionality_error = check_functionality(source, mock_task)
            if not functionality_valid:
                logger.warning(f"Genome failed functionality check: {functionality_error}")
                self._record_violation("functionality_error", source, functionality_error)
                self.evaluation_cache[cache_key] = False
                return False
            
            # Step 3: Parse AST for ethical evaluation
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                logger.warning(f"Failed to parse AST for ethical evaluation: {str(e)}")
                self._record_violation("ast_parse_error", source, str(e))
                self.evaluation_cache[cache_key] = False
                return False
            
            # Step 4: Evaluate ethics
            try:
                ethics_report = evaluate_ethics(source, tree)
                parsed_report = parse_ethics_report(ethics_report)
                
                # Log the ethics evaluation
                logger.info(f"Ethics evaluation completed with score: {parsed_report['overall_score']}")
                for category, score in parsed_report['category_scores'].items():
                    logger.info(f"Ethics category '{category}' score: {score}")
                
                # Check if the code passes the ethics evaluation
                if not parsed_report['passed']:
                    logger.warning(f"Genome failed ethics evaluation with score: {parsed_report['overall_score']}")
                    for concern in parsed_report['concerns']:
                        logger.warning(f"Ethics concern: {concern['category']} - {concern['description']} (Severity: {concern['severity']})")
                    
                    self._record_violation("ethics_violation", source, f"Failed ethics evaluation with score: {parsed_report['overall_score']}")
                    self.evaluation_cache[cache_key] = False
                    return False
                
                # Step 5: Check Gurbani alignment
                is_aligned, alignment_score, concerns = check_gurbani_alignment(parsed_report)
                if not is_aligned:
                    logger.warning(f"Genome failed Gurbani alignment check with score: {alignment_score}")
                    for concern in concerns:
                        logger.warning(f"Gurbani alignment concern: {concern}")
                    
                    self._record_violation("gurbani_alignment", source, f"Failed Gurbani alignment with score: {alignment_score}")
                    self.evaluation_cache[cache_key] = False
                    return False
                
                # All checks passed
                logger.info("Genome passed all post-evolution ethical checks")
                self.evaluation_cache[cache_key] = True
                return True
                
            except Exception as e:
                logger.error(f"Error during ethics evaluation: {str(e)}")
                self._record_violation("ethics_evaluation_error", source, str(e))
                self.evaluation_cache[cache_key] = False
                return False
                
        except Exception as e:
            logger.error(f"Error during post-evolution evaluation: {str(e)}")
            return False
    
    def _get_cache_key(self, source: str) -> str:
        """
        Generate a cache key for a source code string.
        
        Args:
            source: Source code string
            
        Returns:
            Cache key as a string
        """
        return hashlib.md5(source.encode()).hexdigest()
    
    def _record_violation(self, violation_type: str, source: str, details: str) -> None:
        """
        Record an ethical violation for logging and tracking.
        
        Args:
            violation_type: Type of violation
            source: Source code that caused the violation
            details: Details about the violation
        """
        self.violation_history.append({
            'type': violation_type,
            'details': details,
            'timestamp': time.time(),
            'source_excerpt': source[:200] + ('...' if len(source) > 200 else '')
        })
    
    def clear_evaluation_cache(self) -> None:
        """Clear the evaluation cache."""
        self.evaluation_cache = {}
        logger.info("Evaluation cache cleared")
    
    def get_evaluation_cache_size(self) -> int:
        """
        Get the current size of the evaluation cache.
        
        Returns:
            Number of entries in the cache
        """
        return len(self.evaluation_cache)
    
    def add_boundary(self, boundary_name: str, **kwargs) -> None:
        """
        Add an ethical boundary check to the enforcer.
        
        Args:
            boundary_name: Name of the boundary to add
            **kwargs: Parameters for the boundary check
        """
        if self.use_post_evolution:
            logger.warning("Adding pre-evolution boundary while in post-evolution mode")
            
        if boundary_name in self._standard_boundaries:
            self.boundaries[boundary_name] = {
                'check': self._standard_boundaries[boundary_name],
                'params': kwargs
            }
            # Update configuration to reflect this boundary
            if not hasattr(self.config.ethical_boundaries, 'boundaries'):
                self.config.ethical_boundaries.boundaries = {}
            self.config.ethical_boundaries.boundaries[boundary_name] = kwargs
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
        if self.use_post_evolution:
            logger.warning("Adding custom pre-evolution boundary while in post-evolution mode")
            
        self.boundaries[name] = {
            'check': check_function,
            'params': kwargs
        }
        logger.info(f"Added custom boundary: {name}")
    
    def check_all_boundaries(self, genome) -> bool:
        """
        Check if a genome satisfies all defined ethical boundaries.
        
        This is the legacy pre-evolution approach.
        
        Args:
            genome: The genome to check
            
        Returns:
            True if all boundaries are satisfied, False otherwise
        """
        if self.use_post_evolution:
            logger.warning("Using pre-evolution boundary checks while in post-evolution mode")
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
            r'\bos\.spawn',
            r'\bptyprocess\.',
            r'\bos\.fork',
            r'\bos\.execv',
            r'\bos\.execl',
            r'\bplatform\.system',
            r'\bos\.startfile',
            r'\bctypes\.windll',
            r'\bwinreg\.',  # Windows registry access
            r'\bmsvcrt\.',  # Windows-specific system access
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"System call attempt detected: {pattern}")
                return False
        
        # Also check for AST nodes that might indicate system calls
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Check for attribute access that might be system calls
                if isinstance(node, ast.Attribute):
                    attr_name = node.attr.lower()
                    if any(dangerous in attr_name for dangerous in ['system', 'exec', 'spawn', 'popen', 'call']):
                        logger.warning(f"Potential system call via attribute: {attr_name}")
                        return False
        except SyntaxError:
            # If we can't parse the code, fail closed for safety
            logger.warning("Syntax error while checking for system calls")
            return False
            
        return True
    
    def _check_no_eval_exec(self, source: str, **kwargs) -> bool:
        """Check that the code doesn't use eval() or exec()."""
        patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bcompile\s*\([^)]*\bexec\b',
            r'__import__\s*\(',
            r'globals\(\)\s*\[\s*[\'"][^\'"]+[\'"]\s*\]\s*\(',  # Dynamic function calls via globals
            r'locals\(\)\s*\[\s*[\'"][^\'"]+[\'"]\s*\]\s*\(',   # Dynamic function calls via locals
            r'getattr\s*\([^,]+,\s*[\'"]__call__[\'"]\s*\)',    # Getting callable attributes
            r'ast\.literal_eval\s*\(',                          # Even ast.literal_eval can be dangerous in some contexts
            r'pickle\.loads\s*\(',                              # Pickle deserialization can execute code
            r'marshal\.loads\s*\(',                             # Marshal deserialization can execute code
            r'importlib\.',                                     # Dynamic imports
            r'runpy\.',                                         # Running Python modules
            r'code\.',                                          # Interactive interpreter
            r'codeop\.',                                        # Code compilation
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"Code execution attempt detected: {pattern}")
                return False
        
        # Check for string formatting that might be used for code execution
        if re.search(r'f[\'"].*\{.*\}.*[\'"]', source) and re.search(r'\beval\b|\bexec\b', source):
            logger.warning("Potential code execution via f-strings detected")
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
                r'\bpathlib\.Path',
                r'\bos\.path\.exists',
                r'\bos\.access',
                r'\bos\.chmod',
                r'\bos\.chown',
                r'\bos\.link',
                r'\bos\.symlink',
                r'\bos\.truncate',
                r'\bos\.utime',
                r'\bos\.mkdir',
                r'\bos\.rmdir',
                r'\bos\.scandir',
                r'\bos\.walk',
                r'\bglob\.',
                r'\bfileinput\.',
                r'\bfilecmp\.',
                r'\bstat\.',
                r'\bfcntl\.',
                r'\bmmap\.',
                r'\bio\.',
                r'\btempfile\.',
                r'\bcodecs\.open',
            ]
            
            for pattern in patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    logger.warning(f"File operation detected: {pattern}")
                    return False
            
            return True
        else:
            # Implement more sophisticated check for allowed directories
            # Parse the AST to extract file paths and check them
            try:
                tree = ast.parse(source)
                
                # Track potential file paths in string literals
                file_paths = []
                
                for node in ast.walk(tree):
                    # Check for string literals that might be file paths
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        potential_path = node.value
                        # Simple heuristic: strings that look like file paths
                        if ('/' in potential_path or '\\' in potential_path) and '.' in potential_path:
                            file_paths.append(potential_path)
                    
                    # Check for open() calls
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
                        if node.args:
                            # The first argument to open() is the file path
                            arg = node.args[0]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                file_path = arg.value
                                # Check if this path is within allowed directories
                                if not any(file_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                                    logger.warning(f"File operation outside allowed directories: {file_path}")
                                    return False
                
                # Check all potential file paths
                for file_path in file_paths:
                    if os.path.isabs(file_path) and not any(file_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                        logger.warning(f"Potential file operation outside allowed directories: {file_path}")
                        return False
                
                return True
            except SyntaxError:
                # If we can't parse the code, fail closed for safety
                logger.warning("Syntax error while checking file operations")
                return False
    
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
            r'\bsmtplib\.',
            r'\bpoplib\.',
            r'\bimaplib\.',
            r'\btelnetlib\.',
            r'\bnntplib\.',
            r'\bsmtpd\.',
            r'\baiohttp\.',
            r'\basyncio\.open_connection',
            r'\bselectors\.',
            r'\bselect\.',
            r'\bpycurl\.',
            r'\bparamiko\.',
            r'\btwisted\.',
            r'\btornado\.',
            r'\bwebsockets\.',
            r'\bgrpc\.',
            r'\bpika\.',
            r'\bzmq\.',
            r'\bpyzmq\.',
            r'\bsocketserver\.',
            r'\bsocketio\.',
            r'\bssl\.',
            r'\bopenssl\.',
            r'\bcryptography\.hazmat\.primitives\.asymmetric',
            r'\bpyopenssl\.',
            r'\bssh\.',
            r'\bdns\.',
            r'\bdnspython\.',
            r'\bipaddress\.',
            r'\bnetifaces\.',
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"Network access attempt detected: {pattern}")
                return False
        
        # Check for common network port numbers in the code
        port_pattern = r'(?:^|[^0-9])(?:80|443|21|22|23|25|110|143|3306|5432|27017|6379|9200|9300)(?:[^0-9]|$)'
        if re.search(port_pattern, source):
            # If there are port numbers, check if they're in a comment
            for match in re.finditer(port_pattern, source):
                line_start = source.rfind('\n', 0, match.start()) + 1
                line_end = source.find('\n', match.end())
                if line_end == -1:
                    line_end = len(source)
                line = source[line_start:line_end]
                
                # If the port is not in a comment, flag it
                if not re.match(r'^\s*#', line) and not '//' in line[:match.start() - line_start]:
                    logger.warning(f"Potential network port usage detected: {match.group(0)}")
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
            # Use allowed imports from configuration
            allowed_imports = self.config.ethical_boundaries.allowed_imports
        
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
            
            # Also check for dynamic imports via __import__ or importlib
            if re.search(r'__import__\s*\(|importlib\.|__import', source, re.IGNORECASE):
                logger.warning("Potential dynamic import detected")
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
            r'while\s+[^:]+\s*:(?!.*(?:break|return|exit|quit))',  # While loops without break conditions
            r'def\s+[^(]+\([^)]*\):[^#]*\1\s*\(',  # Recursive function calls without base case
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                # Check if there's a timeout mechanism
                if not re.search(r'time\.(sleep|time)\s*\(', source, re.IGNORECASE):
                    logger.warning(f"Potential infinite loop without timeout")
                    return False
        
        # Check for nested loops which might indicate O(n²) or worse complexity
        try:
            tree = ast.parse(source)
            
            class LoopNestingVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.max_loop_nesting = 0
                    self.current_loop_nesting = 0
                
                def visit_For(self, node):
                    self.current_loop_nesting += 1
                    if self.current_loop_nesting > self.max_loop_nesting:
                        self.max_loop_nesting = self.current_loop_nesting
                    self.generic_visit(node)
                    self.current_loop_nesting -= 1
                
                def visit_While(self, node):
                    self.current_loop_nesting += 1
                    if self.current_loop_nesting > self.max_loop_nesting:
                        self.max_loop_nesting = self.current_loop_nesting
                    self.generic_visit(node)
                    self.current_loop_nesting -= 1
            
            visitor = LoopNestingVisitor()
            visitor.visit(tree)
            
            # Flag code with deeply nested loops (potential performance issue)
            if visitor.max_loop_nesting > 3:
                # Check if there's a timeout mechanism for deeply nested loops
                if not re.search(r'time\.(sleep|time)\s*\(', source, re.IGNORECASE):
                    logger.warning(f"Deeply nested loops ({visitor.max_loop_nesting} levels) without timeout")
                    return False
        except SyntaxError:
            # If we can't parse the code, fail closed for safety
            logger.warning("Syntax error while checking execution time")
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
            r'\[\s*[^\]]+\s+for\s+[^\]]+\s+in\s+range\s*\(\s*[0-9]{6,}\s*\)\s*\]',  # Large list comprehensions with range
            r'\{\s*[^}]+\s+for\s+[^}]+\s+in\s+range\s*\(\s*[0-9]{6,}\s*\)\s*\}',  # Large set/dict comprehensions with range
            r'np\.zeros\s*\(\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)\s*\)',  # Large numpy arrays
            r'np\.ones\s*\(\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)\s*\)',  # Large numpy arrays
            r'np\.empty\s*\(\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)\s*\)',  # Large numpy arrays
            r'np\.array\s*\(\s*\[\s*[0-9]{4,}\s*\*\s*\[',  # Large numpy arrays
            r'torch\.zeros\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)',  # Large torch tensors
            r'torch\.ones\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)',  # Large torch tensors
            r'torch\.empty\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)',  # Large torch tensors
            r'pd\.DataFrame\s*\(\s*np\.random\.rand\s*\(\s*[0-9]{4,}\s*,\s*[0-9]{4,}\s*\)\s*\)',  # Large pandas DataFrames
        ]
        
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"Potential excessive memory usage: {pattern}")
                return False
        
        # Check for memory leaks in loops
        try:
            tree = ast.parse(source)
            
            # Look for large container creation inside loops
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    # Check if there are large container creations inside the loop
                    for child in ast.walk(node):
                        if isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp)):
                            # If there's a large comprehension inside a loop, flag it
                            try:
                                if re.search(r'range\s*\(\s*[0-9]{4,}\s*\)', ast.unparse(child), re.IGNORECASE):
                                    logger.warning("Large container creation inside loop")
                                    return False
                            except AttributeError:
                                # ast.unparse might not be available in older Python versions
                                pass
                        
                        # Check for large list/dict/set creation
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            if child.func.id in ['list', 'dict', 'set'] and child.args:
                                try:
                                    # If there's a large container creation inside a loop, flag it
                                    if re.search(r'range\s*\(\s*[0-9]{4,}\s*\)', ast.unparse(child), re.IGNORECASE):
                                        logger.warning("Large container creation inside loop")
                                        return False
                                except AttributeError:
                                    # ast.unparse might not be available in older Python versions
                                    pass
        except (SyntaxError, AttributeError):
            # If we can't parse the code or ast.unparse is not available, fail closed for safety
            logger.warning("Error while checking memory usage")
            return False
        
        return True
    
    def _check_sandboxed_execution(self, source: str, **kwargs) -> bool:
        """
        Check if code attempts to break out of sandboxed execution environment.
        
        This check looks for patterns that might indicate attempts to escape
        the sandbox or access restricted resources.
        """
        # Check for attempts to access system information or environment
        restricted_patterns = [
            # System information and environment
            r'\bos\.environ\b', r'\bplatform\.uname\b', r'\bsys\.platform\b',
            r'\bsys\.version\b', r'\bsys\.path\b', r'\bsys\.modules\b',
            
            # Reflection and introspection
            r'\bsys\._getframe\b', r'\btraceback\.', r'\binspect\.',
            r'\bgc\.', r'\bctypes\.',
            
            # System configuration
            r'\bsys\.settrace\b', r'\bsys\.setprofile\b', r'\bsys\.setrecursionlimit\b',
            
            # Threading and multiprocessing
            r'\bthread\.', r'\bthreading\.', r'\bmultiprocessing\.',
            r'\bconcurrent\.', r'\bsignal\.',
            
            # System resources
            r'\bresource\.', r'\brlimit\.',
            
            # Security-sensitive modules
            r'\bpwd\.', r'\bgrp\.', r'\bcrypt\.',
            r'\bshadow\.', r'\bspwd\.',
            
            # Terminal access
            r'\btermios\.', r'\btty\.', r'\bpty\.',
            r'\bcurses\.',
            
            # Low-level file operations
            r'\bfcntl\.', r'\bmmap\.',
            
            # Network-related
            r'\bselect\.', r'\bpoll\.', r'\bepoll\.',
            r'\bkqueue\.', r'\bdevpoll\.',
            
            # Cryptography
            r'\bcrypto\.', r'\bopenssl\.', r'\bpycrypto\.',
            r'\bcryptography\.', r'\bhashlib\.', r'\bhmac\.',
            
            # Random number generation (potential for manipulation)
            r'\bsecrets\.',
            
            # Testing frameworks (potential for code execution)
            r'\bpytest\.', r'\bunittest\.', r'\bdoctest\.',
            
            # Profiling and debugging
            r'\bprofile\.', r'\bcProfile\.', r'\bpstats\.',
            r'\btimeit\.', r'\btrace\.'
        ]
        
        for pattern in restricted_patterns:
            if re.search(pattern, source, re.IGNORECASE):
                logger.warning(f"Sandbox escape attempt detected: {pattern}")
                return False
        
        # Check for attempts to modify Python's internal state
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Check for attribute assignments that might modify Python's state
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id in ['sys', 'os', 'builtins']:
                                logger.warning(f"Attempt to modify Python internal state: {target.value.id}.{target.attr}")
                                return False
        except SyntaxError:
            # If we can't parse the code, fail closed for safety
            logger.warning("Syntax error while checking for sandbox escape")
            return False
        
        return True
        
    def update_config(self, config: BaseConfig) -> None:
        """
        Update the enforcer's configuration.
        
        Args:
            config: New configuration object
        """
        self.config = config
        
        # Update parameters from configuration
        self.use_post_evolution = self.config.ethical_boundaries.use_post_evolution
        
        # Update boundaries from configuration if they exist
        if hasattr(self.config.ethical_boundaries, 'boundaries') and self.config.ethical_boundaries.boundaries:
            for boundary_name, params in self.config.ethical_boundaries.boundaries.items():
                if boundary_name in self._standard_boundaries:
                    self.boundaries[boundary_name] = {
                        'check': self._standard_boundaries[boundary_name],
                        'params': params
                    }
