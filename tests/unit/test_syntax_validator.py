"""
Unit tests for the SyntaxValidator class.
"""

import unittest
from trisolaris.core.syntax_validator import SyntaxValidator

class TestSyntaxValidator(unittest.TestCase):
    """Test cases for the SyntaxValidator class."""
    
    def test_validate_valid_code(self):
        """Test validation of syntactically valid code."""
        code = """
def hello_world():
    print("Hello, world!")
    return 42
"""
        is_valid, error_msg, _ = SyntaxValidator.validate(code)
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)
    
    def test_validate_invalid_code(self):
        """Test validation of syntactically invalid code."""
        code = """
def hello_world()
    print("Hello, world!")
    return 42
"""
        is_valid, error_msg, _ = SyntaxValidator.validate(code)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
    
    def test_repair_string_literals(self):
        """Test repair of newlines in string literals."""
        code = """
def hello_world():
    print("
Hello, world!")
    return 42
"""
        repaired_code, repairs = SyntaxValidator.repair(code)
        self.assertEqual(len(repairs), 1)
        is_valid, _, _ = SyntaxValidator.validate(repaired_code)
        self.assertTrue(is_valid)
    
    def test_repair_indentation(self):
        """Test repair of indentation issues."""
        code = """
def hello_world():
print("Hello, world!")
return 42
"""
        repaired_code, repairs = SyntaxValidator.repair(code)
        self.assertEqual(len(repairs), 1)
        is_valid, _, _ = SyntaxValidator.validate(repaired_code)
        self.assertTrue(is_valid)
    
    def test_repair_missing_colons(self):
        """Test repair of missing colons in compound statements."""
        code = """
def hello_world()
    print("Hello, world!")
    return 42
"""
        repaired_code, repairs = SyntaxValidator.repair(code)
        self.assertEqual(len(repairs), 1)
        is_valid, _, _ = SyntaxValidator.validate(repaired_code)
        self.assertTrue(is_valid)
    
    def test_repair_unbalanced_delimiters(self):
        """Test repair of unbalanced delimiters."""
        code = """
def hello_world():
    data = {"key": "value"
    print("Hello, world!")
    return 42
"""
        repaired_code, repairs = SyntaxValidator.repair(code)
        self.assertEqual(len(repairs), 1)
        is_valid, _, _ = SyntaxValidator.validate(repaired_code)
        self.assertTrue(is_valid)
    
    def test_validate_and_repair(self):
        """Test the combined validate_and_repair method."""
        code = """
def hello_world():
    print("
Hello, world!")
    if True
        return 42
"""
        repaired_code, was_valid, repairs = SyntaxValidator.validate_and_repair(code)
        self.assertFalse(was_valid)
        self.assertGreater(len(repairs), 0)
        is_valid, _, _ = SyntaxValidator.validate(repaired_code)
        self.assertTrue(is_valid)
    
    def test_multiple_issues(self):
        """Test repair of code with multiple syntax issues."""
        code = """
def hello_world()
    print("
Hello, world!")
    data = {"key": "value"
    if True
        return 42
"""
        repaired_code, was_valid, repairs = SyntaxValidator.validate_and_repair(code)
        self.assertFalse(was_valid)
        self.assertGreater(len(repairs), 1)
        is_valid, _, _ = SyntaxValidator.validate(repaired_code)
        self.assertTrue(is_valid)

if __name__ == '__main__':
    unittest.main()