#!/usr/bin/env python3
"""
Test runner script for the TRISOLARIS framework.

This script runs all tests in the test suite and generates a coverage report.
"""
import os
import sys
import unittest
import argparse

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def discover_and_run_tests(test_type="all", verbose=False):
    """
    Discover and run tests of the specified type.
    
    Args:
        test_type: Type of tests to run (unit, integration, functional, or all)
        verbose: Whether to show verbose output
    
    Returns:
        Test result object
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Set verbosity level
    verbosity = 2 if verbose else 1
    
    # Get absolute path to tests directory
    tests_dir = os.path.join(project_root, "tests")
    
    # Discover tests based on type
    if test_type in ["unit", "all"]:
        unit_tests = loader.discover(os.path.join(tests_dir, "unit"), pattern="test_*.py")
        suite.addTests(unit_tests)
        print(f"Discovered {unit_tests.countTestCases()} unit tests")
    
    if test_type in ["integration", "all"]:
        integration_tests = loader.discover(os.path.join(tests_dir, "integration"), pattern="test_*.py")
        suite.addTests(integration_tests)
        print(f"Discovered {integration_tests.countTestCases()} integration tests")
    
    if test_type in ["functional", "all"]:
        functional_tests = loader.discover(os.path.join(tests_dir, "functional"), pattern="test_*.py")
        suite.addTests(functional_tests)
        print(f"Discovered {functional_tests.countTestCases()} functional tests")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(
        description="Run TRISOLARIS tests"
    )
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "functional", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    # Run with coverage if requested
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage(
                source=["trisolaris"],
                omit=["*/tests/*", "*/examples/*"]
            )
            cov.start()
            
            # Run tests
            print("Running tests with coverage...")
            result = discover_and_run_tests(args.type, args.verbose)
            
            # Generate coverage report
            cov.stop()
            cov.save()
            
            print("\nCoverage Summary:")
            cov.report()
            
            # Generate HTML report
            cov.html_report(directory="coverage_html")
            print("\nHTML coverage report generated in coverage_html/")
            
        except ImportError:
            print("Coverage package not installed. Run 'pip install coverage' to enable coverage reports.")
            # Run tests without coverage
            result = discover_and_run_tests(args.type, args.verbose)
    else:
        # Run tests without coverage
        result = discover_and_run_tests(args.type, args.verbose)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main()) 