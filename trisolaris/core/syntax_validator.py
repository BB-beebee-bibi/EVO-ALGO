"""
Syntax Validator for the TRISOLARIS framework.

This module provides utilities to validate and repair Python code syntax
during the evolutionary process, ensuring that genetic operations produce
syntactically valid code.
"""

import ast
import re
import logging
from typing import Tuple, Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntaxValidator:
    """
    Validates and repairs Python code syntax.
    
    This class provides methods to check if code is syntactically valid
    and to repair common syntax issues that may arise during evolution.
    """
    
    @staticmethod
    def validate(code: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate that the given code has valid Python syntax.
        
        Args:
            code: The source code to validate
            
        Returns:
            Tuple containing:
                - Boolean indicating if syntax is valid
                - Error message if syntax is invalid, None otherwise
                - Error details dictionary if available, None otherwise
        """
        try:
            ast.parse(code)
            return True, None, None
        except SyntaxError as e:
            error_details = {
                'line': e.lineno,
                'column': e.offset,
                'text': e.text,
                'msg': e.msg
            }
            error_msg = f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
            logger.debug(f"Syntax validation failed: {error_msg}")
            return False, error_msg, error_details
        except Exception as e:
            error_msg = f"Unexpected error during syntax validation: {str(e)}"
            logger.debug(error_msg)
            return False, error_msg, None
    
    @staticmethod
    def repair(code: str) -> Tuple[str, List[str]]:
        """
        Attempt to repair common syntax issues in the code.
        
        Args:
            code: The source code to repair
            
        Returns:
            Tuple containing:
                - Repaired code (or original if repair failed)
                - List of applied repairs
        """
        original_code = code
        applied_repairs = []
        
        # Fix 1: Repair newlines in string literals
        code, repair_count = SyntaxValidator._repair_string_literals(code)
        if repair_count > 0:
            applied_repairs.append(f"Fixed {repair_count} newline(s) in string literals")
        
        # Fix 2: Fix indentation issues
        code, repair_count = SyntaxValidator._repair_indentation(code)
        if repair_count > 0:
            applied_repairs.append(f"Fixed {repair_count} indentation issue(s)")
        
        # Fix 3: Fix unbalanced parentheses, brackets, and braces
        code, repair_count = SyntaxValidator._repair_unbalanced_delimiters(code)
        if repair_count > 0:
            applied_repairs.append(f"Fixed {repair_count} unbalanced delimiter(s)")
        
        # Fix 4: Fix missing colons in compound statements
        code, repair_count = SyntaxValidator._repair_missing_colons(code)
        if repair_count > 0:
            applied_repairs.append(f"Fixed {repair_count} missing colon(s)")
        
        # Validate the repaired code
        is_valid, _, _ = SyntaxValidator.validate(code)
        if not is_valid:
            # If repair failed, return the original code
            logger.debug("Repair attempts failed to fix syntax issues")
            return original_code, []
        
        return code, applied_repairs
    
    @staticmethod
    def _repair_string_literals(code: str) -> Tuple[str, int]:
        """
        Repair newlines in string literals.
        
        Args:
            code: The source code to repair
            
        Returns:
            Tuple containing:
                - Repaired code
                - Number of repairs made
        """
        # Pattern to find string literals with newlines
        pattern = r'print\s*\(\s*"(\s*)\n([^"]*?)"\s*\)'
        
        # Replace with proper escaped newlines
        repaired_code, count = re.subn(pattern, r'print("\n\2")', code)
        
        return repaired_code, count
    
    @staticmethod
    def _repair_indentation(code: str) -> Tuple[str, int]:
        """
        Repair indentation issues.
        
        Args:
            code: The source code to repair
            
        Returns:
            Tuple containing:
                - Repaired code
                - Number of repairs made
        """
        lines = code.split('\n')
        repaired_lines = []
        repair_count = 0
        
        # Track indentation level
        indent_level = 0
        indent_size = 4  # Standard Python indentation
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                repaired_lines.append(line)
                continue
            
            # Decrease indent level for lines that end blocks
            if stripped.startswith(('elif', 'else:', 'except:', 'finally:', 'except ', 'finally ')):
                indent_level = max(0, indent_level - 1)
            
            # Calculate expected indentation
            expected_indent = ' ' * (indent_level * indent_size)
            
            # Check if indentation is correct
            current_indent = line[:len(line) - len(line.lstrip())]
            if len(current_indent) != indent_level * indent_size:
                # Fix indentation
                repaired_lines.append(expected_indent + stripped)
                repair_count += 1
            else:
                repaired_lines.append(line)
            
            # Increase indent level for lines that start blocks
            if stripped.endswith(':'):
                indent_level += 1
            
            # Decrease indent level for lines that end blocks
            if stripped == 'return' or stripped.startswith('return '):
                indent_level = max(0, indent_level - 1)
        
        return '\n'.join(repaired_lines), repair_count
    
    @staticmethod
    def _repair_unbalanced_delimiters(code: str) -> Tuple[str, int]:
        """
        Repair unbalanced parentheses, brackets, and braces.
        
        Args:
            code: The source code to repair
            
        Returns:
            Tuple containing:
                - Repaired code
                - Number of repairs made
        """
        # Define delimiter pairs
        delimiters = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        
        repair_count = 0
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Skip comments and docstrings
            if line.strip().startswith('#') or line.strip().startswith("'''") or line.strip().startswith('"""'):
                continue
            
            # Check for unbalanced delimiters in this line
            stack = []
            for char in line:
                if char in delimiters:
                    stack.append(char)
                elif char in delimiters.values():
                    if not stack or delimiters[stack.pop()] != char:
                        # Unbalanced closing delimiter, ignore it
                        pass
            
            # Add missing closing delimiters
            if stack:
                closing = ''.join(delimiters[char] for char in reversed(stack))
                lines[i] = line + closing
                repair_count += len(stack)
        
        return '\n'.join(lines), repair_count
    
    @staticmethod
    def _repair_missing_colons(code: str) -> Tuple[str, int]:
        """
        Repair missing colons in compound statements.
        
        Args:
            code: The source code to repair
            
        Returns:
            Tuple containing:
                - Repaired code
                - Number of repairs made
        """
        # Pattern to find compound statements without colons
        patterns = [
            (r'^\s*(if\s+.*?)\s*$', r'\1:'),
            (r'^\s*(elif\s+.*?)\s*$', r'\1:'),
            (r'^\s*(else)\s*$', r'\1:'),
            (r'^\s*(for\s+.*?)\s*$', r'\1:'),
            (r'^\s*(while\s+.*?)\s*$', r'\1:'),
            (r'^\s*(def\s+.*?\))\s*$', r'\1:'),
            (r'^\s*(class\s+.*?)\s*$', r'\1:'),
            (r'^\s*(try)\s*$', r'\1:'),
            (r'^\s*(except\s+.*?)\s*$', r'\1:'),
            (r'^\s*(except)\s*$', r'\1:'),
            (r'^\s*(finally)\s*$', r'\1:')
        ]
        
        lines = code.split('\n')
        repair_count = 0
        
        for i, line in enumerate(lines):
            for pattern, replacement in patterns:
                if re.match(pattern, line):
                    lines[i] = re.sub(pattern, replacement, line)
                    repair_count += 1
                    break
        
        return '\n'.join(lines), repair_count

    @staticmethod
    def validate_and_repair(code: str) -> Tuple[str, bool, List[str]]:
        """
        Validate the code and repair it if necessary.
        
        Args:
            code: The source code to validate and repair
            
        Returns:
            Tuple containing:
                - Valid code (original or repaired)
                - Boolean indicating if the code was valid initially
                - List of applied repairs
        """
        # First check if the code is already valid
        is_valid, _, _ = SyntaxValidator.validate(code)
        
        if is_valid:
            return code, True, []
        
        # If not valid, attempt to repair
        repaired_code, applied_repairs = SyntaxValidator.repair(code)
        
        # Validate the repaired code
        is_repaired_valid, _, _ = SyntaxValidator.validate(repaired_code)
        
        if is_repaired_valid:
            return repaired_code, False, applied_repairs
        else:
            # If repair failed, return the original code
            return code, False, []