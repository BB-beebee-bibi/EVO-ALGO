"""
Drive Scanner Task for TRISOLARIS

This module implements the Drive Scanner task interface for the TRISOLARIS
evolutionary framework.
"""

import os
import json
import subprocess
import tempfile
import time
import ast
import re
import datetime
from typing import Dict, Any, Tuple, List, Optional

from trisolaris.tasks.base import TaskInterface, TrisolarisBoundary

class DriveScannerTask(TaskInterface):
    """
    Task interface for evolving a drive scanner program.
    
    This implementation uses the DriveScannerFitnessEvaluator to evaluate
    the fitness of evolved drive scanner programs.
    """
    
    def __init__(self, template_path: str = None, test_mountpoint: str = None):
        """
        Initialize the drive scanner task.
        
        Args:
            template_path: Path to the template drive scanner program file
            test_mountpoint: Optional specific mountpoint to use for testing
        """
        self.template_path = template_path
        self.test_mountpoint = test_mountpoint
        
        # If no specific mountpoint is provided, try to find a suitable one
        if not self.test_mountpoint:
            self.test_mountpoint = self._find_test_mountpoint()
    
    def get_name(self) -> str:
        """
        Get the name of this task.
        
        Returns:
            A string identifying this task
        """
        return "drive_scanner"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this task.
        
        Returns:
            A string describing the purpose and functionality of this task
        """
        return (
            "A program that scans all connected storage devices, lets the user "
            "choose one, and creates a detailed snapshot of the selected drive's contents."
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
"""
    
    def evaluate_fitness(self, source_code: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the fitness of the provided source code for this task.
        
        Args:
            source_code: The drive scanner source code to evaluate
            
        Returns:
            A tuple containing (fitness_score, detailed_results)
        """
        results = {
            "syntax_valid": False,
            "runtime_successful": False,
            "execution_time": 0,
            "drive_detection": 0,
            "drive_selection": 0,
            "scan_functionality": 0,
            "analysis_quality": 0,
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
            "drive_detection": 0.15,
            "drive_selection": 0.1,
            "scan_functionality": 0.15,
            "analysis_quality": 0.15,
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
            "drive_detection": 0,
            "drive_selection": 0,
            "scan_functionality": 0,
            "analysis_quality": 0,
            "error_handling": 0,
            "resource_efficiency": 0,
            "user_interface": 0
        }
        
        # Check for drive detection functionality
        drive_detection_patterns = [
            r'lsblk',
            r'subprocess.*check_output',
            r'get_drives',
            r'find.*drives'
        ]
        if any(re.search(pattern, code, re.IGNORECASE) for pattern in drive_detection_patterns):
            results["drive_detection"] = 1.0
        
        # Check for drive selection functionality
        drive_selection_patterns = [
            r'input\s*\(',
            r'select.*drive',
            r'choice',
            r'user.*select'
        ]
        if any(re.search(pattern, code, re.IGNORECASE) for pattern in drive_selection_patterns):
            results["drive_selection"] = 1.0
        
        # Check for scan functionality
        scan_patterns = [
            r'os\.walk',
            r'os\.listdir',
            r'scan_directory',
            r'recursiv',
            r'file_info'
        ]
        scan_score = sum(1 for pattern in scan_patterns if re.search(pattern, code, re.IGNORECASE)) / len(scan_patterns)
        results["scan_functionality"] = min(1.0, scan_score)
        
        # Check for analysis functionality
        analysis_patterns = [
            r'file_types',
            r'analysis',
            r'statistics',
            r'largest_files',
            r'newest_files',
            r'calculate.*size'
        ]
        analysis_score = sum(1 for pattern in analysis_patterns if re.search(pattern, code, re.IGNORECASE)) / len(analysis_patterns)
        results["analysis_quality"] = min(1.0, analysis_score)
        
        # Check for error handling
        error_handling_patterns = [
            r'try\s*:.*except',
            r'error',
            r'exception'
        ]
        if any(re.search(pattern, code, re.IGNORECASE | re.DOTALL) for pattern in error_handling_patterns):
            results["error_handling"] = 1.0
        
        # Check for resource efficiency
        resource_efficiency_patterns = [
            r'max_depth',
            r'limit',
            r'throttle',
            r'generator'
        ]
        resource_score = sum(0.25 for pattern in resource_efficiency_patterns if re.search(pattern, code, re.IGNORECASE))
        results["resource_efficiency"] = min(1.0, resource_score)
        
        # Check for user interface elements
        ui_patterns = [
            r'print\s*\(',
            r'prompt',
            r'input\s*\(',
            r'display',
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
                [os.sys.executable, script_path],
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
                    # Exited cleanly (might have found no drives)
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
    
    def _find_test_mountpoint(self) -> str:
        """Find a suitable test mountpoint."""
        try:
            # Try to find a removable drive first
            output = subprocess.check_output(
                ["lsblk", "-J", "-o", "NAME,TYPE,MOUNTPOINT,HOTPLUG"],
                universal_newlines=True
            )
            
            lsblk_data = json.loads(output)
            
            # Look for mounted partitions on removable drives
            for device in lsblk_data.get("blockdevices", []):
                # Check if this is a removable device
                is_removable = device.get("hotplug") == "1"
                
                if "children" in device:
                    for partition in device["children"]:
                        mountpoint = partition.get("mountpoint")
                        
                        if mountpoint and is_removable:
                            return mountpoint
            
            # If no removable drive is found, fall back to /tmp or /home
            if os.path.exists("/tmp") and os.access("/tmp", os.R_OK | os.W_OK):
                return "/tmp"
            elif os.path.exists("/home") and os.access("/home", os.R_OK | os.W_OK):
                return "/home"
            
        except Exception as e:
            print(f"Error finding test mountpoint: {e}")
        
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
            TrisolarisBoundary.MAX_MEMORY_USAGE: {"max_memory_usage": 100}
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """
        Get the weights for different fitness components.
        
        Returns:
            A dictionary mapping fitness component names to their weights
        """
        return {
            "functionality": 0.7,  # Highest priority for functionality
            "efficiency": 0.2,     # Second priority is efficiency
            "alignment": 0.1       # Ethical alignment is still important
        }
    
    def get_allowed_imports(self) -> List[str]:
        """
        Get the list of allowed imports for this task.
        
        Returns:
            A list of allowed import module names
        """
        return [
            "os", "sys", "json", "datetime", "subprocess", "stat", "shutil", 
            "collections", "tempfile", "time", "re", "logging", "ast"
        ]
    
    def get_evolution_params(self) -> Dict[str, Any]:
        """
        Get recommended evolution parameters for this task.
        
        Returns:
            A dictionary of parameters for the evolution process
        """
        return {
            "population_size": 10,
            "num_generations": 5,
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
