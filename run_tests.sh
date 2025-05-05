#!/bin/bash
# Test runner script for TRISOLARIS

echo "TRISOLARIS Test Runner"
echo "======================"

# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install coverage pytest
else
    source venv/bin/activate
fi

# Create a temporary directory for test artifacts
mkdir -p test_output

# Run unit tests
echo -e "\nRunning unit tests..."
python -m pytest tests/unit -v

# Run integration tests
echo -e "\nRunning integration tests..."
python -m pytest tests/integration -v

# Run functional tests
echo -e "\nRunning functional tests..."
python -m pytest tests/functional -v

# Generate coverage report
echo -e "\nRunning tests with coverage..."
coverage run --source=trisolaris tests/run_tests.py
coverage report -m
coverage html -d test_output/coverage_html

echo -e "\nTests complete!"
echo "Coverage report available in test_output/coverage_html/index.html" 