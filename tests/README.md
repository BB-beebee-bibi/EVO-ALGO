# TRISOLARIS Test Suite

This is the test suite for the TRISOLARIS evolutionary framework. It contains unit, integration, and functional tests for the core components and tasks.

## Structure

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_code_genome.py      # Tests for CodeGenome class
│   └── test_evolution_engine.py # Tests for EvolutionEngine class
│
├── integration/        # Integration tests for component combinations
│   └── test_bluetooth_scanner_task.py # Tests for BluetoothScannerTask
│
├── functional/         # Functional tests for complete features
│   └── test_bluetooth_vulnerability_detection.py # Tests for vulnerability detection
│
├── run_tests.py        # Python test runner
└── README.md           # This file
```

## Running Tests

### Using the Python Script

To run all tests using the Python script:

```bash
python tests/run_tests.py
```

You can also run a specific type of test:

```bash
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type functional
```

For verbose output, add the `-v` flag:

```bash
python tests/run_tests.py -v
```

### Using the Shell Script

Alternatively, you can use the shell script:

```bash
./run_tests.sh
```

This will set up a virtual environment, run all tests, and generate a coverage report.

## Coverage Reports

To generate a coverage report:

```bash
python tests/run_tests.py --coverage
```

This will create an HTML coverage report in the `coverage_html` directory.

## Writing Tests

### Unit Tests

Unit tests should test a single component in isolation. They should be fast and have no external dependencies.

```python
class TestComponent(unittest.TestCase):
    def test_feature(self):
        # Test a specific feature of the component
        component = Component()
        result = component.feature()
        self.assertEqual(result, expected_result)
```

### Integration Tests

Integration tests should test how components work together. They may have some external dependencies but should still be relatively fast.

```python
class TestIntegration(unittest.TestCase):
    def test_component_interaction(self):
        # Test how components interact
        component1 = Component1()
        component2 = Component2(component1)
        result = component2.use_component1()
        self.assertTrue(result)
```

### Functional Tests

Functional tests should test complete features from the user's perspective. They may be slower and depend on external resources.

```python
class TestFeature(unittest.TestCase):
    def test_end_to_end(self):
        # Test a complete feature
        result = run_feature()
        self.assertEqual(result, expected_result)
```

## Adding New Tests

When adding new tests:

1. Place them in the appropriate directory (unit, integration, or functional)
2. Follow the naming convention: `test_*.py`
3. Include a class that inherits from `unittest.TestCase`
4. Name test methods starting with `test_`
5. Include a docstring explaining what the test does
6. Run the tests to ensure they pass 