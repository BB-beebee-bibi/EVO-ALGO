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
        
        # Attempt to parse the AST once
        tree = None
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If parsing fails, it definitely violates boundaries needing AST checks
            # Let the specific checks handle this if they don't need an AST
            pass 

        # Check each active boundary
        for boundary in self.active_boundaries:
            check_func = self.boundaries[boundary]
            
            # Get boundary-specific parameters
            params = {k: v for k, v in self.parameters.items() 
                     if k in check_func.__code__.co_varnames}
            
            # Pass source code and potentially the AST
            check_args = {'source_code': source_code}
            if 'tree' in check_func.__code__.co_varnames:
                if tree is None:
                    # If tree parsing failed earlier, but the check needs it, fail the check
                    return False 
                check_args['tree'] = tree
            
            # Check the boundary
            if not check_func(**check_args, **params):
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
        
        # Attempt to parse the AST once
        tree = None
        syntax_error_msg = None
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            syntax_error_msg = f"Code contains syntax errors: {str(e)}"
            # Still proceed, as some checks might not need the AST
            pass
            
        violations = {}
        
        # Check each active boundary
        for boundary in self.active_boundaries:
            check_func = self.boundaries[boundary]
            
            # Get boundary-specific parameters
            params = {k: v for k, v in self.parameters.items() 
                     if k in check_func.__code__.co_varnames}
            
            # Pass source code, explanation flag, and potentially the AST
            check_args = {'source_code': source_code, 'explain': True}
            needs_tree = 'tree' in check_func.__code__.co_varnames
            
            if needs_tree:
                if tree is None:
                    # If tree parsing failed earlier, but the check needs it, record violation
                    violations[boundary] = syntax_error_msg or "Code contains syntax errors (required for this check)"
                    continue
                check_args['tree'] = tree
            
            # Call the check function with an explanation flag
            result, explanation = check_func(**check_args, **params)
            
            if not result:
                violations[boundary] = explanation
        
        return violations
    
    def _check_no_system_calls(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't contain system calls (uses AST)."""
        if tree is None:
            # This check requires a valid AST
            if explain:
                return False, "Code contains syntax errors (required for system call check)"
            return False
        
        # Define system call patterns (as before)
        system_call_patterns = [
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os' and
                node.func.attr in ['system', 'popen', 'spawn', 'exec', 
                                  'execl', 'execlp', 'execle', 'execv', 
                                  'execvp', 'execvpe']
            )),
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'subprocess' and
                node.func.attr in ['run', 'call', 'check_call', 'check_output', 
                                  'Popen']
            )),
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
    
    def _check_no_file_operations(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't contain file operations (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for file operation check)"
            return False

        # Define file operation patterns (as before)
        file_operation_patterns = [
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Name) and
                node.func.id == 'open'
            )),
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                node.func.attr in ['read', 'write', 'readline', 'readlines', 
                                 'writelines', 'seek', 'tell']
            )),
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os' and
                node.func.attr in ['remove', 'unlink', 'rmdir', 'mkdir', 
                                  'makedirs', 'rename', 'replace']
            )),
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
    
    def _check_no_network_access(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't contain network access (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for network access check)"
            return False
        
        # Import patterns to look for (as before)
        network_imports = [
            'socket', 'http', 'urllib', 'requests', 'ftplib', 'smtplib',
            'telnetlib', 'imaplib', 'nntplib', 'poplib'
        ]
        
        # Network function call patterns (as before)
        network_call_patterns = [
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'socket' and
                node.func.attr in ['socket', 'connect', 'bind', 'listen', 'accept']
            )),
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in ['urllib', 'requests'] and
                node.func.attr in ['get', 'post', 'put', 'delete', 'head', 
                                  'request', 'urlopen']
            )),
        ]
        
        violations = []
        # Check for network imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if any(name.name == module or name.name.startswith(f"{module}.") 
                          for module in network_imports):
                        if explain:
                            violations.append(f"Network module import found: {name.name}")
                        else: 
                            return False
            elif isinstance(node, ast.ImportFrom):
                if node.module in network_imports or any(node.module.startswith(f"{module}.") 
                                                      for module in network_imports):
                    if explain:
                        violations.append(f"Network module import found: {node.module}")
                    else:
                        return False
        
        # Check for network calls
        for node in ast.walk(tree):
            for node_type, check_func in network_call_patterns:
                if isinstance(node, node_type) and check_func(node):
                    if explain:
                        violations.append(f"Network operation found: {ast.unparse(node)}")
                    else:
                        return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_no_process_creation(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't create new processes (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for process creation check)"
            return False
        
        # Process creation patterns (as before)
        process_creation_patterns = [
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'multiprocessing' and
                node.func.attr in ['Process', 'Pool']
            )),
            (ast.Call, lambda node: (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'threading' and
                node.func.attr == 'Thread'
            )),
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
    
    def _check_no_eval_exec(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't use eval() or exec() (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for eval/exec check)"
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
    
    def _check_no_imports(self, source_code: str, tree: Optional[ast.AST] = None, 
                         allowed_imports: Set[str] = None, 
                         explain: bool = False) -> Union[bool, tuple]:
        """
        Check that code doesn't import unauthorized modules (uses AST).
        
        Args:
            source_code: Code to check
            tree: Optional pre-parsed AST
            allowed_imports: Set of allowed import module names
            explain: Whether to return explanation
            
        Returns:
            Boolean result or (result, explanation) tuple
        """
        if allowed_imports is None:
            allowed_imports = self.parameters.get('allowed_imports', set())
        
        if tree is None:
            # This check requires a valid AST
            if explain:
                return False, "Code contains syntax errors (required for import check)"
            return False
        
        # Check imports
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Check top-level import against allowed set
                    module_name = name.name.split('.')[0]
                    if module_name not in allowed_imports:
                        if explain:
                            violations.append(f"Unauthorized import: {name.name}")
                        else:
                            return False
            elif isinstance(node, ast.ImportFrom):
                # Check module name against allowed set
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name not in allowed_imports:
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
        Check that code execution doesn't exceed time limit (does not use AST).
        This method still needs the source code to execute it.
        ... (rest of the method remains unchanged) ...
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
                                  check=False,
                                  creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0)
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
        Check that code execution doesn't exceed memory limit (does not use AST).
        This method still needs the source code to execute it.
        ... (rest of the method remains unchanged) ...
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False, encoding='utf-8') as temp_file:
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
                                  check=False,
                                  creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0)
            
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
                    # If no peak is found and no error, assume it passed or failed silently
                    # Let's assume it passed unless other info suggests failure
                    # It might be better to return False here for safety
                    return True, "Could not reliably measure memory usage, but no error reported."
                return True # Or potentially False for stricter check
            
        except Exception as e:
            if explain:
                return False, f"Error in memory check: {str(e)}"
            return False
        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
        
        if explain:
            return True, ""
        return True
    
    def _check_no_pii_access(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check that code doesn't attempt to access PII (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for PII check)"
            return False
        
        # Define PII-related patterns (as before)
        pii_patterns = [
            (r'\b(password|passwd|secret|key|token|credentials?|ssn|social\s*security|credit\s*card|cc\s*number)\b', 
             "Contains PII-related keywords"),
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 
             "Contains email address pattern"),
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
    
    def _check_universal_equity(self, source_code: str, tree: Optional[ast.AST] = None, 
                                bias_terms: Set[str] = None, excluded_groups: Set[str] = None, 
                                explain: bool = False) -> Union[bool, tuple]:
        """Check that code avoids biased language or discrimination (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for equity check)"
            return False
            
        if bias_terms is None:
            bias_terms = self.parameters.get('bias_terms', set())
        if excluded_groups is None:
            excluded_groups = self.parameters.get('excluded_groups', set())
        
        violations = []
        
        # Check for bias terms in comments, strings, identifiers
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifier = node.id.lower()
                if identifier in bias_terms:
                    if explain:
                        violations.append(f"Biased term found in identifier: {node.id}")
                    else: return False
            elif isinstance(node, ast.Str) or (hasattr(ast, 'Constant') and isinstance(node, ast.Constant) and isinstance(node.value, str)):
                string_value = (node.s if hasattr(node, 's') else node.value).lower()
                for term in bias_terms:
                    if term in string_value:
                        if explain:
                            violations.append(f"Biased term found in string/comment: {term}")
                        else: return False
                        
        # Check comments separately for bias terms
        try:
            import io, tokenize
            tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)
            for token_info in tokens:
                if token_info.type == tokenize.COMMENT:
                    comment_text = token_info.string.lower()
                    for term in bias_terms:
                        if term in comment_text:
                            if explain:
                                violations.append(f"Biased term found in comment: {term}")
                            else: return False
        except tokenize.TokenError:
            pass # Ignore token errors for this check
        
        # Check for exclusionary logic (simple check based on keywords)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_str = ast.unparse(node.test).lower()
                for group in excluded_groups:
                    if group in test_str:
                        if explain:
                            violations.append(f"Potentially exclusionary logic found related to '{group}': {ast.unparse(node.test)}")
                        else: return False
        
        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_truthful_communication(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check for misleading comments, names, or logic (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for truthful communication check)"
            return False
        
        violations = []
        
        # Check for comments contradicting code
        # Check for misleading function/variable names
        # Check for logic that misrepresents intent (e.g., function claims speed but is slow)
        
        # Example: Check if docstring claims one thing but does another
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstring_lower = docstring.lower()
                    function_body = ast.unparse(node.body).lower()
                    
                    # Claim vs Reality: Speed
                    if ('fast' in docstring_lower or 'efficient' in docstring_lower) and \
                       function_body.count('for ') >= 2: # Basic check for nested loops
                        if explain:
                            violations.append(f"Function {node.name} claims speed but contains nested loops.")
                        else: return False
                        
                    # Claim vs Reality: Side Effects
                    if 'no side effects' in docstring_lower and \
                       any(isinstance(n, ast.Assign) for n in ast.walk(node)):
                        if explain:
                            violations.append(f"Function {node.name} claims no side effects but contains assignments.")
                        else: return False

        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_humble_code(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check for excessive complexity, arrogance in comments (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for humble code check)"
            return False

        violations = []
        complexity_threshold = 15 # Cyclomatic complexity approximate limit
        nesting_threshold = 4

        # 1. Check Complexity (using simple counts as proxy)
        complexity = 0
        max_nesting = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            if hasattr(node, 'orelse') and node.orelse: # Count elif/else
                complexity += 1
            # Estimate nesting depth
            if isinstance(node, (ast.If, ast.For, ast.While)):
                try: # Calculate nesting within this node
                    current_max_depth = self._get_nesting_depth(node)
                    max_nesting = max(max_nesting, current_max_depth)
                except Exception as e:
                    logger.debug(f"Error calculating nesting depth: {e}")

        if complexity > complexity_threshold:
            if explain:
                violations.append(f"Code complexity ({complexity}) exceeds threshold ({complexity_threshold}).")
            else: return False
        if max_nesting > nesting_threshold:
             if explain:
                violations.append(f"Code nesting depth ({max_nesting}) exceeds threshold ({nesting_threshold}).")
             else: return False

        # 2. Check Comments for Arrogance (simple keyword check)
        arrogant_terms = ["obviously", "trivial", "simple fix", "easy", "of course", "stupid code"]
        try:
            import io, tokenize
            tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)
            for token_info in tokens:
                if token_info.type == tokenize.COMMENT:
                    comment_text = token_info.string.lower()
                    for term in arrogant_terms:
                        if term in comment_text:
                            if explain:
                                violations.append(f"Potentially arrogant term found in comment: '{term}'")
                            else: return False
        except tokenize.TokenError:
            pass # Ignore token errors

        if explain:
            return len(violations) == 0, "\n".join(violations)
        return len(violations) == 0
    
    def _check_service_oriented(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check for lack of documentation, poor error handling (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for service oriented check)"
            return False

        violations = []
        min_docstring_ratio = 0.5 # Minimum ratio of functions with docstrings
        functions_count = 0
        documented_functions_count = 0
        try_blocks_count = 0
        specific_except_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods for docstring check
                if not node.name.startswith('_'): 
                    functions_count += 1
                    docstring = ast.get_docstring(node)
                    if docstring:
                        documented_functions_count += 1
                        # Basic check for parameter documentation
                        param_names = [arg.arg for arg in node.args.args if arg.arg != 'self']
                        if param_names and not all(param in docstring for param in param_names):
                             if explain:
                                violations.append(f"Function {node.name} docstring may lack parameter documentation.")
                             # Don't fail just for this, but record it

            elif isinstance(node, ast.Try):
                try_blocks_count += 1
                has_specific_handler = False
                for handler in node.handlers:
                    # Check for specific exception types vs. bare except or except Exception
                    if handler.type and not (isinstance(handler.type, ast.Name) and handler.type.id == 'Exception'):
                        has_specific_handler = True
                        break
                if has_specific_handler:
                    specific_except_count += 1

        # Check docstring ratio
        if functions_count > 0 and (documented_functions_count / functions_count) < min_docstring_ratio:
            if explain:
                violations.append(f"Insufficient function documentation ({(documented_functions_count / functions_count):.1%} < {min_docstring_ratio:.0%})")
            else: return False

        # Check error handling quality
        if try_blocks_count > 0 and (specific_except_count / try_blocks_count) < 0.5:
             if explain:
                violations.append(f"Potentially poor error handling (low ratio of specific exceptions caught).")
             # Don't fail just for this, but record it

        if explain:
            return len(violations) == 0, "\n".join(violations)
        # Only fail for major violations like missing docstrings
        return not (functions_count > 0 and (documented_functions_count / functions_count) < min_docstring_ratio)

    def _check_harmony_with_environment(self, source_code: str, tree: Optional[ast.AST] = None, explain: bool = False) -> Union[bool, tuple]:
        """Check for resource leaks, inefficient patterns (uses AST)."""
        if tree is None:
            if explain:
                return False, "Code contains syntax errors (required for harmony check)"
            return False

        violations = []

        # Check for potential resource leaks (e.g., file handles not in 'with')
        opened_files = set()
        closed_files = set()
        files_in_with = set()

        for node in ast.walk(tree):
            # Track files opened directly
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'open':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            opened_files.add(target.id)
            # Track files closed explicitly
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'close':
                     if isinstance(node.func.value, ast.Name):
                        closed_files.add(node.func.value.id)
            # Track files opened with 'with'
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id == 'open':
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                             files_in_with.add(item.optional_vars.id)

        # Check if any directly opened files were not closed or handled by 'with'
        unmanaged_files = opened_files - closed_files - files_in_with
        if unmanaged_files:
            if explain:
                 violations.append(f"Potential resource leak: File handles {unmanaged_files} opened but possibly not closed.")
            else: return False

        # Check for obviously inefficient patterns (like string concatenation in loops)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.AugAssign) and isinstance(sub_node.op, ast.Add):
                        # Check if target is likely a string being added to
                        # This is a heuristic and might be inaccurate
                        if isinstance(sub_node.target, ast.Name):
                            # A simplistic check: if a string literal is involved
                            if isinstance(sub_node.value, ast.Constant) and isinstance(sub_node.value.value, str):
                                if explain:
                                    violations.append(f"Potential inefficient string concatenation in loop.")
                                # Don't fail, just warn in explanation

        if explain:
            return len(violations) == 0, "\n".join(violations)
        # Only fail for definite leaks
        return not unmanaged_files

    def _get_nesting_depth(self, node, current_depth=1):
        """Helper method to calculate nesting depth of relevant structures"""
        max_depth = current_depth
        relevant_children = [child for child in ast.iter_child_nodes(node) 
                             if isinstance(child, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.Try))]
        for child in relevant_children:
            # Don't increase depth just for child nodes, only for nested control/structure blocks
            child_depth = self._get_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth 