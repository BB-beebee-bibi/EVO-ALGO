"""
Ethical Boundary Enforcer for the TRISOLARIS framework.

This module implements hard constraints for code evaluation, ensuring solutions
adhere to ethical boundaries and safety standards.
"""

import ast
import re
import os
import subprocess
import tempfile
import threading
import time
import platform
from typing import List, Dict, Set, Optional, Any, Union, Callable

# Platform-specific imports
if platform.system() != 'Windows':
    import resource  # Unix-only module
else:
    # Windows alternative for resource monitoring
    import psutil

class EthicalBoundaryEnforcer:
    """
    Implements hard ethical and safety constraints for code evaluation.
    
    This class provides a framework for checking code solutions against various
    ethical and safety constraints before they undergo fitness evaluation.
    """
    
    def __init__(self):
        """Initialize the Ethical Boundary Enforcer with default boundaries."""
        # Dictionary of boundary checking functions
        self.boundaries = {
            # System boundaries
            "no_system_calls": self._check_no_system_calls,
            "no_file_operations": self._check_no_file_operations,
            "no_network_access": self._check_no_network_access,
            "no_process_creation": self._check_no_process_creation,
            "no_imports": self._check_no_imports,
            
            # Resource boundaries
            "max_execution_time": self._check_execution_time,
            "max_memory_usage": self._check_memory_usage,
            
            # Privacy boundaries
            "no_pii_access": self._check_no_pii_access,
            
            # Security boundaries
            "no_eval_exec": self._check_no_eval_exec,
            
            # Universal harmony boundaries
            "universal_equity": self._check_universal_equity,
            "truthful_communication": self._check_truthful_communication,
            "humble_code": self._check_humble_code,
            "service_oriented": self._check_service_oriented,
            "harmony_with_environment": self._check_harmony_with_environment
        }
        
        # Boundary parameters
        self.parameters = {
            "max_execution_time": 1.0,  # seconds
            "max_memory_usage": 100,    # MB
            "allowed_imports": set(),   # empty set means all imports are blocked
            "bias_terms": set([
                "superior", "inferior", "better than", "worse than",
                "master", "slave", "blacklist", "whitelist",
                "dominate", "control", "exploit"
            ]),
            "excluded_groups": set([
                "gender", "race", "nationality", "ethnicity", "religion",
                "age", "disability", "orientation", "identity", "class"
            ])
        }
        
        # Active boundaries
        self.active_boundaries = set()
    
    def add_boundary(self, boundary_name: str, **kwargs):
        """
        Activate a boundary with optional parameters.
        
        Args:
            boundary_name: Name of the boundary to activate
            **kwargs: Optional parameters for the boundary
        """
        if boundary_name not in self.boundaries:
            raise ValueError(f"Unknown boundary: {boundary_name}")
        
        self.active_boundaries.add(boundary_name)
        
        # Update parameters if provided
        for param, value in kwargs.items():
            if param in self.parameters:
                self.parameters[param] = value
    
    def remove_boundary(self, boundary_name: str):
        """
        Deactivate a boundary.
        
        Args:
            boundary_name: Name of the boundary to deactivate
        """
        if boundary_name in self.active_boundaries:
            self.active_boundaries.remove(boundary_name)
    
    def check(self, genome: Any) -> bool:
        """
        Check if a genome passes all active ethical boundaries.
        
        Args:
            genome: The code genome to check
            
        Returns:
            True if all boundaries pass, False otherwise
        """
        # Get the source code from the genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        # Check each active boundary
        for boundary in self.active_boundaries:
            check_func = self.boundaries[boundary]
            
            # Get boundary-specific parameters
            params = {k: v for k, v in self.parameters.items() 
                     if k in check_func.__code__.co_varnames}
            
            # Check the boundary
            if not check_func(source_code, **params):
                return False
        
        return True
    
    def explain_violations(self, genome: Any) -> Dict[str, str]:
        """
        Check a genome against all active boundaries and return explanations for violations.
        
        Args:
            genome: The code genome to check
            
        Returns:
            Dictionary mapping boundary names to violation explanations
        """
        # Get the source code from the genome
        if hasattr(genome, 'to_source'):
            source_code = genome.to_source()
        else:
            source_code = str(genome)
        
        violations = {}
        
        # Check each active boundary
        for boundary in self.active_boundaries:
            check_func = self.boundaries[boundary]
            
            # Get boundary-specific parameters
            params = {k: v for k, v in self.parameters.items() 
                     if k in check_func.__code__.co_varnames}
            
            # Call the check function with an explanation flag
            params['explain'] = True
            result, explanation = check_func(source_code, **params)
            
            if not result:
                violations[boundary] = explanation
        
        return violations
    
    def _check_no_system_calls(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't contain system calls."""
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Define system call patterns
        system_call_patterns = [
            # os module system calls
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os' and
                node.func.attr in ['system', 'popen', 'spawn', 'exec', 
                                  'execl', 'execlp', 'execle', 'execv', 
                                  'execvp', 'execvpe']
            )),
            # subprocess module calls
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'subprocess' and
                node.func.attr in ['run', 'call', 'check_call', 'check_output', 
                                  'Popen']
            )),
            # direct calls to system
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Name) and
                node.func.id in ['system', 'popen', 'execl', 'execlp', 'execle',
                               'execv', 'execvp', 'execvpe']
            )),
        ]
        
        # Check for system calls
        violations = []
        for node in ast.walk(tree):
            for node_type, check_func in system_call_patterns:
                if isinstance(node, node_type) and check_func(node):
                    if explain:
                        violations.append(f"System call found: {ast.unparse(node)}")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_no_file_operations(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't contain file operations."""
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Define file operation patterns
        file_operation_patterns = [
            # open() function
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Name) and
                node.func.id == 'open'
            )),
            # file methods
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                node.func.attr in ['read', 'write', 'readline', 'readlines', 
                                 'writelines', 'seek', 'tell']
            )),
            # os file operations
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os' and
                node.func.attr in ['remove', 'unlink', 'rmdir', 'mkdir', 
                                  'makedirs', 'rename', 'replace']
            )),
            # with open as pattern
            (ast.With, lambda node: any(
                isinstance(item.context_expr, ast.Call) and
                isinstance(item.context_expr.func, ast.Name) and
                item.context_expr.func.id == 'open'
                for item in node.items
            )),
        ]
        
        # Check for file operations
        violations = []
        for node in ast.walk(tree):
            for node_type, check_func in file_operation_patterns:
                if isinstance(node, node_type) and check_func(node):
                    if explain:
                        violations.append(f"File operation found: {ast.unparse(node)}")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_no_network_access(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't contain network access."""
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Import patterns to look for
        network_imports = [
            'socket', 'http', 'urllib', 'requests', 'ftplib', 'smtplib',
            'telnetlib', 'imaplib', 'nntplib', 'poplib'
        ]
        
        # Network function call patterns
        network_call_patterns = [
            # socket operations
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'socket' and
                node.func.attr in ['socket', 'connect', 'bind', 'listen', 'accept']
            )),
            # urllib/requests
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in ['urllib', 'requests'] and
                node.func.attr in ['get', 'post', 'put', 'delete', 'head', 
                                  'request', 'urlopen']
            )),
        ]
        
        # Check for network imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if any(name.name == module or name.name.startswith(f"{module}.") 
                          for module in network_imports):
                        if explain:
                            return False, f"Network module import found: {name.name}"
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module in network_imports or any(node.module.startswith(f"{module}.") 
                                                      for module in network_imports):
                    if explain:
                        return False, f"Network module import found: {node.module}"
                    return False
        
        # Check for network calls
        for node in ast.walk(tree):
            for node_type, check_func in network_call_patterns:
                if isinstance(node, node_type) and check_func(node):
                    if explain:
                        return False, f"Network operation found: {ast.unparse(node)}"
                    return False
        
        if explain:
            return True, ""
        return True
    
    def _check_no_process_creation(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't create new processes."""
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Process creation patterns
        process_creation_patterns = [
            # multiprocessing module
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'multiprocessing' and
                node.func.attr in ['Process', 'Pool']
            )),
            # threading module
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'threading' and
                node.func.attr == 'Thread'
            )),
            # concurrent.futures
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Attribute) and
                isinstance(node.func.value.value, ast.Name) and
                node.func.value.value.id == 'concurrent' and
                node.func.value.attr == 'futures' and
                node.func.attr in ['ProcessPoolExecutor', 'ThreadPoolExecutor']
            )),
        ]
        
        # Check for process creation
        violations = []
        for node in ast.walk(tree):
            for node_type, check_func in process_creation_patterns:
                if isinstance(node, node_type) and check_func(node):
                    if explain:
                        violations.append(f"Process creation found: {ast.unparse(node)}")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_no_eval_exec(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't use eval() or exec()."""
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Look for eval or exec
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['eval', 'exec']:
                if explain:
                    violations.append(f"Insecure {node.func.id}() found: {ast.unparse(node)}")
                else:
                    return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_no_imports(self, source_code: str, allowed_imports: Set[str] = None, 
                         explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code doesn't import unauthorized modules.
        
        Args:
            source_code: Code to check
            allowed_imports: Set of allowed import module names
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        if allowed_imports is None:
            allowed_imports = self.parameters.get('allowed_imports', set())
        
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Check imports
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if not any(name.name == module or name.name.startswith(f"{module}.") 
                              for module in allowed_imports):
                        if explain:
                            violations.append(f"Unauthorized import: {name.name}")
                        else:
                            return False
            elif isinstance(node, ast.ImportFrom):
                if not any(node.module == module or node.module.startswith(f"{module}.") 
                          for module in allowed_imports):
                    if explain:
                        violations.append(f"Unauthorized import from: {node.module}")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_execution_time(self, source_code: str, max_execution_time: float = 1.0,
                            explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code execution doesn't exceed time limit.
        
        Args:
            source_code: Code to check
            max_execution_time: Maximum allowed execution time in seconds
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file.write(source_code)
            temp_file_path = temp_file.name
        
        try:
            # Set up a timeout mechanism for execution
            result = {"completed": False, "time": 0}
            
            def run_code():
                start_time = time.time()
                try:
                    subprocess.run(['python', temp_file_path], 
                                  timeout=max_execution_time, 
                                  capture_output=True, 
                                  check=False)
                    result["completed"] = True
                except subprocess.TimeoutExpired:
                    result["completed"] = False
                result["time"] = time.time() - start_time
            
            # Run in a separate thread to handle internal timeouts
            thread = threading.Thread(target=run_code)
            thread.start()
            thread.join(max_execution_time + 0.5)  # Give a little extra time for subprocess timeout
            
            if thread.is_alive():
                if explain:
                    return False, f"Code execution exceeded time limit of {max_execution_time} seconds"
                return False
            
            if not result["completed"] or result["time"] > max_execution_time:
                if explain:
                    return False, f"Code execution time of {result['time']:.2f} seconds exceeded limit of {max_execution_time} seconds"
                return False
            
            if explain:
                return True, f"Code executed in {result['time']:.2f} seconds"
            return True
            
        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _check_memory_usage(self, source_code: str, max_memory_usage: int = 100,
                          explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code execution doesn't exceed memory limit.
        
        Args:
            source_code: Code to check
            max_memory_usage: Maximum allowed memory usage in MB
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            # Indent the source code
            indented_code = "\n".join("    " + line for line in source_code.splitlines())
            
            # Create a wrapper that monitors memory usage
            if platform.system() != 'Windows':
                # Unix version using resource module
                memory_monitoring_code = """
import resource
import sys

max_memory_mb = {0}

def monitored_code():
{1}

peak_memory = 0
try:
    monitored_code()
    # Get peak memory usage
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Convert to MB (different units on different platforms)
    if sys.platform == 'darwin':
        peak_memory = peak_memory / 1024 / 1024  # macOS reports in bytes
    else:
        peak_memory = peak_memory / 1024  # Linux reports in KB
    
    print(f"MEMORY_PEAK: {{peak_memory}}")
except Exception as e:
    print(f"EXECUTION_ERROR: {{str(e)}}")
"""
            else:
                # Windows version using psutil
                memory_monitoring_code = """
import psutil
import os
import sys

max_memory_mb = {0}

def monitored_code():
{1}

peak_memory = 0
try:
    process = psutil.Process(os.getpid())
    monitored_code()
    # Get peak memory usage in MB
    peak_memory = process.memory_info().peak_wset / 1024 / 1024
    print(f"MEMORY_PEAK: {{peak_memory}}")
except Exception as e:
    print(f"EXECUTION_ERROR: {{str(e)}}")
"""
            
            # Format the code with memory limit and the source code
            formatted_code = memory_monitoring_code.format(max_memory_usage, indented_code)
            temp_file.write(formatted_code)
            temp_file_path = temp_file.name
        
        try:
            # Run the monitoring code
            result = subprocess.run(['python', temp_file_path], 
                                  capture_output=True, 
                                  text=True,
                                  check=False)
            
            # Extract peak memory usage
            output = result.stdout + result.stderr
            memory_match = re.search(r'MEMORY_PEAK: ([\d\.]+)', output)
            
            if memory_match:
                peak_memory = float(memory_match.group(1))
                if peak_memory > max_memory_usage:
                    if explain:
                        return False, f"Code exceeded memory limit: {peak_memory:.2f}MB > {max_memory_usage}MB"
                    return False
            else:
                error_match = re.search(r'EXECUTION_ERROR: (.+)', output)
                if error_match:
                    error_msg = error_match.group(1)
                    if explain:
                        return False, f"Error executing code: {error_msg}"
                    return False
                if explain:
                    return False, "Failed to measure memory usage"
                return False
            
        except Exception as e:
            if explain:
                return False, f"Error in memory check: {str(e)}"
            return False
        finally:
            # Clean up
            os.unlink(temp_file_path)
        
        if explain:
            return True, ""
        return True
    
    def _check_no_pii_access(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code doesn't attempt to access personally identifiable information.
        
        Args:
            source_code: Code to check
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Define PII-related patterns
        pii_patterns = [
            # String literals with PII keywords
            (r'\b(password|passwd|secret|key|token|credentials?|ssn|social\s*security|credit\s*card|cc\s*number)\b', 
             "Contains PII-related keywords"),
            
            # Email regex pattern
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 
             "Contains email address pattern"),
            
            # Phone number pattern
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 
             "Contains phone number pattern"),
        ]
        
        # Check string literals for PII patterns
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Str) or (hasattr(ast, 'Constant') and 
                                            isinstance(node, ast.Constant) and 
                                            isinstance(node.value, str)):
                
                # Get the string value
                string_value = node.s if hasattr(node, 's') else node.value
                
                # Check against each pattern
                for pattern, message in pii_patterns:
                    if re.search(pattern, string_value, re.IGNORECASE):
                        if explain:
                            violations.append(f"{message}: {string_value}")
                        else:
                            return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_universal_equity(self, source_code: str, bias_terms: Set[str] = None, 
                               excluded_groups: Set[str] = None, explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code doesn't contain biased or discriminatory language or logic.
        
        Args:
            source_code: Code to check
            bias_terms: Set of terms indicating bias
            excluded_groups: Set of groups that should not be discriminated against
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        if bias_terms is None:
            bias_terms = self.parameters.get('bias_terms', set())
        
        if excluded_groups is None:
            excluded_groups = self.parameters.get('excluded_groups', set())
        
        # Create regex patterns for bias terms
        bias_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) 
                        for term in bias_terms]
        
        # Create regex patterns for excluded groups
        group_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) 
                         for term in excluded_groups]
        
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Check string literals for bias
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Str) or (hasattr(ast, 'Constant') and 
                                            isinstance(node, ast.Constant) and 
                                            isinstance(node.value, str)):
                
                # Get the string value
                string_value = node.s if hasattr(node, 's') else node.value
                
                # Check for bias terms
                for pattern in bias_patterns:
                    if pattern.search(string_value):
                        if explain:
                            violations.append(f"Biased term found: {string_value}")
                        else:
                            return False
                
                # Check for references to excluded groups in a possible discriminatory context
                for pattern in group_patterns:
                    if pattern.search(string_value):
                        # Look for nearby bias terms to determine if it's potentially discriminatory
                        for bias_pattern in bias_patterns:
                            if bias_pattern.search(string_value):
                                if explain:
                                    violations.append(f"Potentially discriminatory content: {string_value}")
                                else:
                                    return False
        
        # Check variable names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                # Check for bias terms in variable names
                for term in bias_terms:
                    if term.lower() in name.lower():
                        if explain:
                            violations.append(f"Biased term in variable name: {name}")
                        else:
                            return False
        
        # Check comments
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if hasattr(child, 'lineno'):
                    line_no = child.lineno
                    if line_no <= len(source_code.splitlines()):
                        line = source_code.splitlines()[line_no - 1]
                        comment_start = line.find('#')
                        if comment_start != -1:
                            comment = line[comment_start + 1:].strip()
                            # Check for bias terms in comments
                            for pattern in bias_patterns:
                                if pattern.search(comment):
                                    if explain:
                                        violations.append(f"Biased term in comment: {comment}")
                                    else:
                                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_truthful_communication(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code doesn't intentionally deceive or mislead.
        
        Args:
            source_code: Code to check
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        # Define deceptive patterns
        deceptive_patterns = [
            # Misleading variable names
            (r'(is|has|should)_[a-z_]+\s*=\s*(False|0)', 
             "Variable name suggests true but value is false"),
            
            # Deliberate confusion
            (r'(True\s*==\s*False|False\s*==\s*True)', 
             "Deliberately confusing boolean comparison"),
            
            # Misleading comments vs code
            (r'#.*correct.*\n.*\breturn False\b', 
             "Comment claims correctness but returns false"),
            
            # Overrides that silently ignore inputs
            (r'def\s+[a-zA-Z_]+\([^)]*\):\s*[^#\n]*pass', 
             "Function appears to do nothing without warning")
        ]
        
        # Check patterns in source code
        violations = []
        source_lines = source_code.splitlines()
        for pattern, message in deceptive_patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            for match in matches:
                if explain:
                    violations.append(f"{message}: {match.group(0)}")
                else:
                    return False
        
        # Look for intentionally confusing comments
        clean_code = re.sub(r'#.*$', '', source_code, flags=re.MULTILINE)
        has_comments = source_code != clean_code
        if has_comments:
            # Function with a docstring claiming one behavior but doing another
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Check if docstring claims to do one thing but return has contrary logic
                        for docstring_word in ["increment", "add", "plus", "increase"]:
                            if docstring_word in docstring.lower():
                                # Look for decreasing operations in function body
                                function_body = ast.unparse(node)
                                if re.search(r'(decrement|subtract|minus|decrease)', function_body, re.IGNORECASE):
                                    if explain:
                                        violations.append(f"Function {node.name} claims to {docstring_word} but does the opposite")
                                    else:
                                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_humble_code(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code follows principles of simplicity and readability.
        
        Args:
            source_code: Code to check
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        violations = []
        
        # Check for overly complex list/dict comprehensions
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                # Count the number of for clauses and if clauses
                num_for = len([c for c in node.generators])
                num_if = sum(len(gen.ifs) for gen in node.generators)
                
                if num_for > 2 or num_if > 2:
                    if explain:
                        violations.append(f"Overly complex comprehension with {num_for} for clauses and {num_if} if clauses")
                    else:
                        return False
            
            # Check for overly complex lambda functions
            if isinstance(node, ast.Lambda):
                lambda_source = ast.unparse(node)
                if len(lambda_source) > 80:
                    if explain:
                        violations.append(f"Overly complex lambda function: {lambda_source[:50]}...")
                    else:
                        return False
            
            # Check for excessive nesting
            if isinstance(node, (ast.If, ast.For, ast.While)):
                max_depth = self._get_nesting_depth(node)
                if max_depth > 4:  # More than 4 levels of nesting is considered excessive
                    if explain:
                        violations.append(f"Excessive nesting depth of {max_depth}")
                    else:
                        return False
        
        # Check for function complexity
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Approximate cyclomatic complexity by counting branches
                complexity = 1  # Base complexity
                for subnode in ast.walk(node):
                    if isinstance(subnode, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                        complexity += 1
                
                if complexity > 10:  # McCabe complexity threshold
                    if explain:
                        violations.append(f"Function {node.name} has high complexity score of {complexity}")
                    else:
                        return False
                
                # Check function length
                function_body = ast.unparse(node)
                num_lines = function_body.count('\n')
                if num_lines > 50:  # More than 50 lines is considered too long
                    if explain:
                        violations.append(f"Function {node.name} is too long ({num_lines} lines)")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _get_nesting_depth(self, node, current_depth=1):
        """Helper method to calculate nesting depth"""
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While)):
                child_depth = self._get_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _check_service_oriented(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code is designed with a service orientation rather than exploitation.
        
        Args:
            source_code: Code to check
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        violations = []
        
        # Check for function documentation
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods (starting with _)
                if node.name.startswith('_'):
                    continue
                    
                # Check for docstring
                docstring = ast.get_docstring(node)
                if not docstring:
                    if explain:
                        violations.append(f"Function {node.name} lacks documentation")
                    else:
                        return False
                
                # Check parameter documentation
                if docstring and node.args.args:
                    # Check if all parameters are documented
                    param_names = [arg.arg for arg in node.args.args if arg.arg != 'self']
                    if param_names:
                        # Simple heuristic: Check if each parameter name appears in the docstring
                        for param in param_names:
                            if param not in docstring:
                                if explain:
                                    violations.append(f"Parameter {param} of function {node.name} is not documented")
                                else:
                                    return False
        
        # Check for excessive resource consumption
        resource_intensive_patterns = [
            (r'while\s+True\s*:', "Infinite loop without clear exit condition"),
            (r'for\s+\w+\s+in\s+range\s*\(\s*[0-9]+\s*\*\s*[0-9]+\s*\)', "Potentially large loop range")
        ]
        
        for pattern, message in resource_intensive_patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            for match in matches:
                # Check if there's a break or return within the loop
                match_pos = match.start()
                indentation = source_code[:match_pos].rfind("\n")
                if indentation == -1:
                    indentation = 0
                else:
                    indentation = match_pos - indentation - 1
                
                # Get the loop's content by finding all lines with greater indentation
                loop_content = ""
                current_pos = match_pos + len(match.group(0))
                for line in source_code[current_pos:].split("\n"):
                    if line.startswith(" " * (indentation + 4)) or line.startswith("\t" * (indentation // 4 + 1)):
                        loop_content += line + "\n"
                    elif line.strip() and not line.isspace():
                        break
                
                # Check if there's an exit condition
                if not re.search(r'(break|return|exit|sys\.exit)', loop_content):
                    if explain:
                        violations.append(f"{message}: {match.group(0)}")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_harmony_with_environment(self, source_code: str, explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code respects computational resources and doesn't needlessly waste them.
        
        Args:
            source_code: Code to check
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        # Parse the AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            if explain:
                return False, "Code contains syntax errors"
            return False
        
        violations = []
        
        # Check for resource leaks
        resource_patterns = [
            # Check for file handles not closed
            (ast.With, lambda node: not any(
                isinstance(item.context_expr, ast.Call) and
                isinstance(item.context_expr.func, ast.Name) and
                item.context_expr.func.id == 'open'
                for item in node.items
            )),
            
            # Check for direct file operations without close
            (ast.Assign, lambda node: (
                isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Name) and
                node.value.func.id == 'open'
            ))
        ]
        
        # Track file handles that are opened
        file_handles = []
        
        # Check for file operations
        for node in ast.walk(tree):
            # Check for 'open' calls without a 'with' context
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and hasattr(node.value.func, 'id') and node.value.func.id == 'open':
                # This is a file being opened directly
                if len(node.targets) > 0 and isinstance(node.targets[0], ast.Name):
                    file_handles.append(node.targets[0].id)
        
        # Now check if all file handles are properly closed
        for handle in file_handles:
            # Look for a close call on this handle
            close_found = False
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and 
                    isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == handle and 
                    node.func.attr == 'close'):
                    close_found = True
                    break
            
            if not close_found:
                if explain:
                    violations.append(f"File handle {handle} is not explicitly closed")
                else:
                    return False
        
        # Check for inefficient algorithms and data structures
        algorithm_patterns = [
            # Nested loops with direct array access (likely O(n²) complexity)
            (r'for\s+\w+\s+in\s+.+:\s*\n\s+for\s+\w+\s+in\s+.+:\s*\n\s+for\s+\w+\s+in\s+.+:', 
             "Triple nested loop detected (potentially O(n³) complexity)"),
            
            # Excessive string concatenation in loops
            (r'for\s+\w+\s+in\s+.+:\s*\n\s+.+\s*\+=\s*str\(', 
             "String concatenation in a loop (consider using join() or a list)"),
             
            # Creation of large temporary lists
            (r'(\[\s*[a-zA-Z0-9_]+\s+for\s+[a-zA-Z0-9_]+\s+in\s+range\s*\(\s*10\d{3,}\s*\)\s*\])',
             "Creation of a very large temporary list")
        ]
        
        for pattern, message in algorithm_patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            for match in matches:
                if explain:
                    violations.append(f"{message}: {match.group(0)}")
                else:
                    return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0 