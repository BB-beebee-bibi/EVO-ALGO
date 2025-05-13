"""
Syntax and Functionality Checker for the TRISOLARIS framework.

This module provides utilities to verify code syntax, functionality, and output
measurement for evolved code. It serves as part of the post-evolution ethical
evaluation system, ensuring that code not only meets ethical standards but also
functions as expected.
"""

import ast
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, Union
import contextlib
import io
import sys
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntaxCheckError(Exception):
    """Exception raised when code fails syntax checking."""
    pass

class FunctionalityCheckError(Exception):
    """Exception raised when code fails functionality checking."""
    pass

class OutputMeasurementError(Exception):
    """Exception raised when code output measurement fails."""
    pass

@lru_cache(maxsize=128)
def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that code parses without syntax errors.
    
    Args:
        code: The source code to check
        
    Returns:
        Tuple containing:
            - Boolean indicating if syntax is valid
            - Error message if syntax is invalid, None otherwise
    """
    try:
        ast.parse(code)
        logger.info("Syntax check passed")
        return True, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
        logger.warning(f"Syntax check failed: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during syntax check: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def check_functionality(code: str, task: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify code performs at least one testable action related to the task.
    
    This function analyzes the AST to determine if the code contains
    operations that would accomplish the specified task.
    
    Args:
        code: The source code to check
        task: Dictionary containing task specifications and requirements
        
    Returns:
        Tuple containing:
            - Boolean indicating if functionality check passed
            - Error message if check failed, None otherwise
    """
    # First ensure syntax is valid
    syntax_valid, error_msg = check_syntax(code)
    if not syntax_valid:
        return False, f"Cannot check functionality due to syntax errors: {error_msg}"
    
    try:
        tree = ast.parse(code)
        
        # Check if code has at least one function definition
        has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        
        # Check if code has at least one function call
        has_call = any(isinstance(node, ast.Call) for node in ast.walk(tree))
        
        # Check if code has any assignment statements
        has_assignment = any(isinstance(node, ast.Assign) for node in ast.walk(tree))
        
        # Check if code has any return statements
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
        
        # Check for task-specific functionality
        task_type = task.get('type', '').lower()
        task_specific_check = False
        
        if task_type == 'network_scanner':
            # Check for network-related operations
            task_specific_check = any(
                isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                any(net_op in node.func.attr for net_op in ['connect', 'socket', 'scan', 'ping'])
                for node in ast.walk(tree)
            )
        elif task_type == 'drive_scanner':
            # Check for file/drive operations
            task_specific_check = any(
                isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                any(file_op in node.func.attr for file_op in ['open', 'read', 'write', 'listdir', 'walk'])
                for node in ast.walk(tree)
            )
        elif task_type == 'bluetooth_scanner':
            # Check for bluetooth operations
            task_specific_check = any(
                isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                any(bt_op in node.func.attr for bt_op in ['discover', 'scan', 'bluetooth', 'ble'])
                for node in ast.walk(tree)
            )
        else:
            # Generic check for any task
            task_specific_check = has_function and has_call
        
        # Code should have at least basic structure and some task-specific functionality
        basic_structure = has_function and (has_call or has_assignment) and has_return
        
        if basic_structure and task_specific_check:
            logger.info("Functionality check passed")
            return True, None
        else:
            missing = []
            if not has_function:
                missing.append("function definitions")
            if not has_call and not has_assignment:
                missing.append("function calls or assignments")
            if not has_return:
                missing.append("return statements")
            if not task_specific_check:
                missing.append(f"task-specific functionality for {task_type}")
            
            error_msg = f"Code lacks required functionality: {', '.join(missing)}"
            logger.warning(f"Functionality check failed: {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error during functionality check: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, error_msg

def measure_output(code: str, task: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Measure and validate output or state changes from code execution.
    
    This function captures stdout/stderr and monitors state changes
    to evaluate if the code produces expected outputs for the given task.
    
    Args:
        code: The source code to measure
        task: Dictionary containing task specifications and expected outputs
        
    Returns:
        Tuple containing:
            - Boolean indicating if output measurement passed
            - Dictionary with measurement results including:
                - stdout: Captured standard output
                - stderr: Captured standard error
                - execution_time: Time taken to execute in seconds
                - success: Whether execution was successful
                - error: Error message if execution failed
    """
    # First ensure syntax is valid
    syntax_valid, error_msg = check_syntax(code)
    if not syntax_valid:
        return False, {
            'stdout': '',
            'stderr': '',
            'execution_time': 0,
            'success': False,
            'error': f"Cannot measure output due to syntax errors: {error_msg}"
        }
    
    # Prepare result dictionary
    result = {
        'stdout': '',
        'stderr': '',
        'execution_time': 0,
        'success': False,
        'error': None
    }
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Compile the code first to catch syntax errors
        compiled_code = compile(code, '<string>', 'exec')
        
        # Create a namespace for execution
        namespace = {}
        
        # Measure execution time
        start_time = time.time()
        
        # Execute with stdout/stderr capture
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(compiled_code, namespace)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Capture output
        result['stdout'] = stdout_capture.getvalue()
        result['stderr'] = stderr_capture.getvalue()
        result['execution_time'] = execution_time
        result['success'] = True
        
        # Check if there's any output
        if not result['stdout'] and not result['stderr']:
            logger.warning("Code executed but produced no output")
            # This is not necessarily a failure, as some valid code might not produce output
        
        logger.info(f"Output measurement completed in {execution_time:.4f} seconds")
        return True, result
        
    except Exception as e:
        result['stderr'] = f"{stderr_capture.getvalue()}\n{traceback.format_exc()}"
        result['error'] = f"Error during code execution: {str(e)}"
        logger.error(f"Output measurement failed: {str(e)}")
        return False, result
    finally:
        stdout_capture.close()
        stderr_capture.close()

def validate_output_against_task(output_result: Dict[str, Any], task: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate measured output against task requirements.
    
    Args:
        output_result: Dictionary with measurement results from measure_output
        task: Dictionary containing task specifications and expected outputs
        
    Returns:
        Tuple containing:
            - Boolean indicating if validation passed
            - Error message if validation failed, None otherwise
    """
    if not output_result['success']:
        return False, f"Cannot validate output: execution failed with error: {output_result['error']}"
    
    # Extract expected outputs from task
    expected_outputs = task.get('expected_outputs', {})
    
    # If no expected outputs defined, consider it a pass
    if not expected_outputs:
        return True, None
    
    # Check for expected strings in stdout
    if 'contains' in expected_outputs:
        for expected_string in expected_outputs['contains']:
            if expected_string not in output_result['stdout']:
                return False, f"Expected output '{expected_string}' not found in stdout"
    
    # Check for unexpected strings in stdout
    if 'excludes' in expected_outputs:
        for unexpected_string in expected_outputs['excludes']:
            if unexpected_string in output_result['stdout']:
                return False, f"Unexpected output '{unexpected_string}' found in stdout"
    
    # Check for maximum execution time
    if 'max_execution_time' in expected_outputs:
        if output_result['execution_time'] > expected_outputs['max_execution_time']:
            return False, f"Execution time ({output_result['execution_time']:.4f}s) exceeds maximum allowed ({expected_outputs['max_execution_time']}s)"
    
    return True, None