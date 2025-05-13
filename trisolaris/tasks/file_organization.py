"""
File Organization Task for TRISOLARIS

This module implements the File Organization task interface for the TRISOLARIS
evolutionary framework, focusing on content categorization of mixed file types.
"""

import os
import json
import subprocess
import tempfile
import time
import ast
import re
import datetime
import hashlib
import mimetypes
import shutil
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional

from trisolaris.tasks.base import TaskInterface, TrisolarisBoundary

class FileOrganizationTask(TaskInterface):
    """
    Task interface for evolving a file organization program.
    
    This implementation focuses on content categorization of mixed file types,
    identifying and grouping similar files based on content and file type,
    with safety measures and content analysis capabilities.
    """
    
    def __init__(self, template_path: str = None, test_directory: str = None):
        """
        Initialize the file organization task.
        
        Args:
            template_path: Path to the template file organization program file
            test_directory: Optional specific directory to use for testing
        """
        self.template_path = template_path
        self.test_directory = test_directory
        
        # If no specific test directory is provided, try to find a suitable one
        if not self.test_directory:
            self.test_directory = self._find_test_directory()
    
    def get_name(self) -> str:
        """
        Get the name of this task.
        
        Returns:
            A string identifying this task
        """
        return "file_organization"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this task.
        
        Returns:
            A string describing the purpose and functionality of this task
        """
        return (
            "A program that analyzes and organizes files based on content similarity, "
            "file type, and other characteristics, with safety measures and content analysis capabilities."
        )
    
    def get_template(self) -> str:
        """
        Get the template code to start evolution from.
        
        Returns:
            A string containing the template source code
        """
        if self.template_path and os.path.exists(self.template_path):
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Return a minimal template as a fallback
            return """#!/usr/bin/env python3
'''
File Organization Tool

This tool analyzes and organizes files based on content similarity,
file type, and other characteristics. It includes safety measures
and content analysis capabilities for mixed file types.
'''

import os
import sys
import hashlib
import shutil
import mimetypes
import re
import json
import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("file_organizer")

def analyze_directory(directory_path: str) -> Dict[str, Any]:
    '''
    Analyze the contents of a directory.
    
    Args:
        directory_path: Path to the directory to analyze
        
    Returns:
        Dictionary with analysis results
    '''
    logger.info(f"Analyzing directory: {directory_path}")
    # TODO: Implement directory analysis
    return {}

def categorize_files(files_info: Dict[str, Any]) -> Dict[str, List[str]]:
    '''
    Categorize files based on content and type.
    
    Args:
        files_info: Dictionary with file information
        
    Returns:
        Dictionary mapping categories to lists of file paths
    '''
    logger.info("Categorizing files by content and type")
    # TODO: Implement file categorization
    return {}

def organize_files(source_dir: str, categories: Dict[str, List[str]], 
                  operation: str = 'copy', confirm: bool = True) -> bool:
    '''
    Organize files according to categories.
    
    Args:
        source_dir: Source directory containing files
        categories: Dictionary mapping categories to lists of file paths
        operation: Operation to perform ('copy' or 'move')
        confirm: Whether to confirm before performing operations
        
    Returns:
        True if successful, False otherwise
    '''
    logger.info(f"Organizing files using {operation} operation")
    # TODO: Implement file organization with safety measures
    return True

def main():
    '''Main function to run the file organizer.'''
    if len(sys.argv) < 2:
        print("Usage: file_organizer.py <directory_path> [--move] [--no-confirm]")
        return
    
    directory_path = sys.argv[1]
    operation = 'move' if '--move' in sys.argv else 'copy'
    confirm = '--no-confirm' not in sys.argv
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    try:
        # Analyze directory contents
        files_info = analyze_directory(directory_path)
        
        # Categorize files
        categories = categorize_files(files_info)
        
        # Organize files
        success = organize_files(directory_path, categories, operation, confirm)
        
        if success:
            print("File organization completed successfully")
        else:
            print("File organization failed or was cancelled")
            
    except Exception as e:
        print(f"Error during file organization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
    
    def evaluate_fitness(self, source_code: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the fitness of the provided source code for this task.
        
        Args:
            source_code: The file organization source code to evaluate
            
        Returns:
            A tuple containing (fitness_score, detailed_results)
        """
        results = {
            "syntax_valid": False,
            "runtime_successful": False,
            "execution_time": 0,
            "content_analysis": 0,
            "file_categorization": 0,
            "similarity_detection": 0,
            "safety_measures": 0,
            "error_handling": 0,
            "resource_efficiency": 0,
            "user_interface": 0,
            "errors": []
        }
        
        # Check syntax validity
        try:
            ast.parse(source_code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["errors"].append(f"Syntax error: {str(e)}")
            # Return early if syntax is invalid
            return 0.0, results
        
        # Create a temporary file for the program
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp_file:
                temp_filename = temp_file.name
                temp_file.write(source_code)
            
            # Make the script executable
            os.chmod(temp_filename, 0o755)
            
            # Check required functionality
            results.update(self._check_required_functionality(source_code))
            
            # Execute the program to assess its runtime behavior
            # This is a mock execution since we can't actually automate user input
            results.update(self._check_execution(temp_filename))
        
        finally:
            # Clean up the temporary file
            if 'temp_filename' in locals():
                try:
                    os.unlink(temp_filename)
                except:
                    pass
        
        # Calculate the overall fitness score
        weights = {
            "syntax_valid": 0.1,
            "runtime_successful": 0.1,
            "content_analysis": 0.15,
            "file_categorization": 0.15,
            "similarity_detection": 0.15,
            "safety_measures": 0.1,
            "error_handling": 0.1,
            "resource_efficiency": 0.05,
            "user_interface": 0.1
        }
        
        fitness_score = 0.0
        for criterion, weight in weights.items():
            score = float(results.get(criterion, 0))
            fitness_score += score * weight
        
        return fitness_score, results
    
    def _check_required_functionality(self, code: str) -> Dict[str, Any]:
        """
        Check if the code includes required functionality.
        
        Args:
            code: Source code to check
            
        Returns:
            Dictionary with functionality scores
        """
        results = {
            "content_analysis": 0,
            "file_categorization": 0,
            "similarity_detection": 0,
            "safety_measures": 0,
            "error_handling": 0,
            "resource_efficiency": 0,
            "user_interface": 0
        }
        
        # Check for content analysis functionality
        content_analysis_patterns = [
            r'mime[_-]?types?',
            r'file[_-]?type',
            r'content[_-]?analysis',
            r'analyze[_-]?content',
            r'read.*file',
            r'open\s*\([^)]*\)\s*\.\s*read'
        ]
        content_score = sum(0.2 for pattern in content_analysis_patterns if re.search(pattern, code, re.IGNORECASE))
        results["content_analysis"] = min(1.0, content_score)
        
        # Check for file categorization functionality
        categorization_patterns = [
            r'categori[zs]e',
            r'group[_-]?by',
            r'classify',
            r'sort[_-]?files',
            r'organize[_-]?by'
        ]
        categorization_score = sum(0.2 for pattern in categorization_patterns if re.search(pattern, code, re.IGNORECASE))
        results["file_categorization"] = min(1.0, categorization_score)
        
        # Check for similarity detection functionality
        similarity_patterns = [
            r'hash',
            r'similar',
            r'duplicate',
            r'compare',
            r'distance',
            r'fingerprint'
        ]
        similarity_score = sum(0.2 for pattern in similarity_patterns if re.search(pattern, code, re.IGNORECASE))
        results["similarity_detection"] = min(1.0, similarity_score)
        
        # Check for safety measures
        safety_patterns = [
            r'confirm',
            r'backup',
            r'safe[_-]?mode',
            r'dry[_-]?run',
            r'copy.*not.*move',
            r'ask.*before'
        ]
        safety_score = sum(0.2 for pattern in safety_patterns if re.search(pattern, code, re.IGNORECASE))
        results["safety_measures"] = min(1.0, safety_score)
        
        # Check for error handling
        error_handling_patterns = [
            r'try\s*:.*except',
            r'error',
            r'exception',
            r'validate',
            r'check.*exist'
        ]
        if any(re.search(pattern, code, re.IGNORECASE | re.DOTALL) for pattern in error_handling_patterns):
            results["error_handling"] = 1.0
        
        # Check for resource efficiency
        resource_efficiency_patterns = [
            r'limit',
            r'throttle',
            r'generator',
            r'yield',
            r'batch'
        ]
        resource_score = sum(0.2 for pattern in resource_efficiency_patterns if re.search(pattern, code, re.IGNORECASE))
        results["resource_efficiency"] = min(1.0, resource_score)
        
        # Check for user interface elements
        ui_patterns = [
            r'print\s*\(',
            r'prompt',
            r'input\s*\(',
            r'display',
            r'progress',
            r'user.*friendly'
        ]
        ui_score = sum(0.2 for pattern in ui_patterns if re.search(pattern, code, re.IGNORECASE))
        results["user_interface"] = min(1.0, ui_score)
        
        return results
    
    def _check_execution(self, script_path: str) -> Dict[str, Any]:
        """
        Check execution behavior of the script.
        
        Args:
            script_path: Path to the script to execute
            
        Returns:
            Dictionary with execution results
        """
        results = {
            "runtime_successful": 0,
            "execution_time": 0
        }
        
        try:
            # Start a process but don't wait for it to complete since it will ask for user input
            # We'll just check if it starts without crashing immediately
            start_time = time.time()
            process = subprocess.Popen(
                [os.sys.executable, script_path, self.test_directory or os.getcwd()],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for a short time to see if the program crashes immediately
            time.sleep(1)
            
            # Check if the process is still running
            if process.poll() is None:
                # Process still running, which is good
                results["runtime_successful"] = 1.0
                
                # Kill the process since we won't be providing input
                process.kill()
            else:
                # Process exited immediately, check return code
                if process.returncode == 0:
                    # Exited cleanly (might have found no files to organize)
                    results["runtime_successful"] = 0.7
                else:
                    # Crashed
                    results["runtime_successful"] = 0.0
                    stdout, stderr = process.communicate()
                    if stderr:
                        results.setdefault("errors", []).append(f"Runtime error: {stderr}")
            
            end_time = time.time()
            results["execution_time"] = end_time - start_time
            
        except Exception as e:
            results["runtime_successful"] = 0.0
            results.setdefault("errors", []).append(f"Execution error: {str(e)}")
        
        return results
    
    def _find_test_directory(self) -> str:
        """Find a suitable test directory with mixed file types."""
        try:
            # Try to use /tmp or a similar directory first
            if os.path.exists("/tmp") and os.access("/tmp", os.R_OK | os.W_OK):
                return "/tmp"
            
            # On Windows, try the temp directory
            if os.name == 'nt' and 'TEMP' in os.environ and os.path.exists(os.environ['TEMP']):
                return os.environ['TEMP']
            
            # Fall back to user's home directory
            home_dir = os.path.expanduser("~")
            if os.path.exists(home_dir) and os.access(home_dir, os.R_OK):
                return home_dir
            
        except Exception as e:
            print(f"Error finding test directory: {e}")
        
        # Last resort: use current directory
        return os.getcwd()
    
    def get_required_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the ethical boundaries required for this task.
        
        Returns:
            A dictionary mapping boundary names to their parameters
        """
        return {
            TrisolarisBoundary.NO_EVAL_EXEC: {},
            TrisolarisBoundary.NO_NETWORK_ACCESS: {},
            TrisolarisBoundary.MAX_EXECUTION_TIME: {"max_execution_time": 5.0},
            TrisolarisBoundary.MAX_MEMORY_USAGE: {"max_memory_usage": 100},
            TrisolarisBoundary.HARMONY_WITH_ENVIRONMENT: {}  # Ensure file operations are respectful
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """
        Get the weights for different fitness components.
        
        Returns:
            A dictionary mapping fitness component names to their weights
        """
        return {
            "functionality": 0.6,  # Highest priority for functionality
            "efficiency": 0.2,     # Second priority is efficiency
            "alignment": 0.2       # Ethical alignment is important for file operations
        }
    
    def get_allowed_imports(self) -> List[str]:
        """
        Get the list of allowed imports for this task.
        
        Returns:
            A list of allowed import module names
        """
        return [
            "os", "sys", "json", "datetime", "subprocess", "stat", "shutil", 
            "collections", "tempfile", "time", "re", "logging", "ast",
            "hashlib", "mimetypes", "pathlib", "filecmp", "difflib"
        ]
    
    def get_evolution_params(self) -> Dict[str, Any]:
        """
        Get recommended evolution parameters for this task.
        
        Returns:
            A dictionary of parameters for the evolution process
        """
        return {
            "population_size": 500,
            "num_generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        }
    
    def post_process(self, source_code: str) -> str:
        """
        Perform post-processing on evolved source code.
        
        Adds a shebang line and ensures the file has proper
        executable permissions.
        
        Args:
            source_code: The evolved source code
            
        Returns:
            The post-processed source code
        """
        # Ensure there's a proper shebang
        if not source_code.startswith("#!/"):
            source_code = "#!/usr/bin/env python3\n" + source_code
        
        # Add a timestamp in a comment
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_code = source_code.replace(
            "#!/usr/bin/env python3\n",
            f"#!/usr/bin/env python3\n# Generated by TRISOLARIS on {timestamp}\n"
        )
        
        return source_code