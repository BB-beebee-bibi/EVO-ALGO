# Progremon Implementation Guidelines

This document provides practical guidelines and best practices for implementing the new version of `progremon.py` based on the architectural plan in `project_summary.md` and the details in `progremon_architecture_detail.md`.

## Implementation Approach

Follow these steps when implementing the rewritten progremon.py file:

1. **Start with the core structure**
   - Set up the import structure
   - Define the Colors and display utilities
   - Implement the ProgemonTrainer class skeleton

2. **Implement the helper classes**
   - TaskTemplateLoader
   - EvolutionSession
   - Any utility functions needed

3. **Implement the main logic**
   - ProgemonTrainer.__init__
   - ProgemonTrainer.main
   - request parsing methods
   - configuration methods

4. **Implement the evolution core**
   - run_evolution method with proper error handling
   - integration with AdaptiveTweaker
   - integration with EthicalBoundaryEnforcer

5. **Add the Bluetooth-specific functionality**
   - template loading
   - parameter configuration
   - output formatting

## Testing Strategy

### Unit Testing

Test each component independently:

```python
# Example unit test for TaskTemplateLoader
def test_task_template_loader():
    loader = TaskTemplateLoader(templates_dir="test_templates")
    template = loader.load_template("bluetooth_scan")
    assert template is not None
    assert "def scan_bluetooth_devices()" in template
    
    # Test non-existent template
    assert loader.load_template("non_existent") is None
```

### Integration Testing

Test the integration between components:

```python
# Example integration test for ProgemonTrainer and AdaptiveTweaker
def test_adaptive_tweaking_integration():
    trainer = ProgemonTrainer()
    config = trainer.process_request("Create a Bluetooth scanner")
    
    # Set up test parameters
    config["gens"] = 2  # Use a small number for quick testing
    config["pop_size"] = 5
    
    # Run evolution
    success = trainer.run_evolution(config)
    assert success
    
    # Verify that adaptive tweaking was applied
    assert len(trainer.adaptive_tweaker.history) > 0
```

### End-to-End Testing

Test the complete process:

```python
def test_end_to_end_evolution():
    trainer = ProgemonTrainer()
    
    # Simulate user input
    request = "Create a Bluetooth scanner that updates every 1 second"
    
    # Process request
    config = trainer.process_request(request)
    settings = trainer.configure_evolution(config)
    
    # Run evolution
    success = trainer.run_evolution(settings)
    assert success
    
    # Verify output files
    session_dir = trainer.session.base_dir
    best_solution_path = session_dir / "best_solution" / "best.py"
    assert best_solution_path.exists()
    
    # Check contents of solution
    with open(best_solution_path, 'r') as f:
        code = f.read()
        assert "bluetooth" in code
        assert "scan" in code
        assert "1 second" in code or "update" in code
```

## Common Integration Issues and Solutions

### 1. AdaptiveTweaker Integration

**Problem:** AdaptiveTweaker methods don't match what's called from ProgemonTrainer.

**Solution:**
- Add a compatibility method in AdaptiveTweaker:
  ```python
  def update_parameters(self, avg_fitness, best_fitness):
      # Create mock population
      mock_population = [...]
      self.record_metrics(mock_population, best_fitness, avg_fitness)
  ```

### 2. Ethical Boundary Enforcement

**Problem:** EthicalBoundaryEnforcer can't access genome.code attribute.

**Solution:**
- Ensure CodeGenome objects expose their source code through a consistent interface:
  ```python
  # In progremon.py
  def _configure_ethical_boundaries(self, enforcer, settings):
      # Set a custom accessor for code extraction
      enforcer.set_code_accessor(lambda genome: genome.to_source())
  ```

### 3. Template Integration

**Problem:** Template code not properly incorporated into evolving solutions.

**Solution:**
- Modify population initialization to include template code:
  ```python
  template_code = self.template_loader.load_template(settings["task"])
  engine.initialize_population(
      size=settings["pop_size"],
      task_description=settings["description"],
      template_code=template_code,
      task_type=settings["task"]
  )
  ```

## Error Handling Best Practices

### 1. Use Specific Exception Types

Instead of catching all exceptions broadly, catch specific ones:

```python
try:
    # Operation that might fail
    pass
except FileNotFoundError as e:
    logging.error(f"Template file not found: {e}")
    # Handle missing file
except PermissionError as e:
    logging.error(f"Permission denied when accessing file: {e}")
    # Handle permission issue
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    logging.error(traceback.format_exc())
    # Handle generic error
```

### 2. Provide Context in Logs

Always include context in log messages:

```python
logging.error(f"Error during generation {generation} evaluation: {str(e)}")
logging.error(f"Population size: {len(population)}, Task type: {settings['task']}")
```

### 3. Use Structured Logging

For complex operations, use structured logging:

```python
logging.info(
    "Generation completed",
    extra={
        "generation": gen,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "valid_solutions": len(valid_scores),
        "mutation_rate": engine.mutation_rate
    }
)
```

### 4. Graceful Recovery

Implement recovery mechanisms for common failures:

```python
def run_evolution(self, settings):
    try:
        # Main evolution code
        pass
    except EvolutionEngineError as e:
        logging.error(f"Engine error: {e}")
        
        # Try to recover
        logging.info("Attempting recovery by reinitializing engine")
        try:
            # Reinitialize engine
            engine = self._initialize_engine(settings)
            # Continue with reduced settings
            settings["gens"] = max(5, settings["gens"] // 2)
            return self._run_evolution_with_engine(engine, settings)
        except Exception as recovery_error:
            logging.error(f"Recovery failed: {recovery_error}")
            return False
```

## Performance Optimization Guidelines

### 1. Lazy Loading

Avoid loading resources until they're needed:

```python
class TaskTemplateLoader:
    def __init__(self, templates_dir):
        self.templates_dir = templates_dir
        self._template_cache = {}
    
    def load_template(self, task_type):
        # Load from cache if available
        if task_type in self._template_cache:
            return self._template_cache[task_type]
            
        # Otherwise load from disk and cache
        template_file = os.path.join(self.templates_dir, f"{task_type}_template.py")
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                template = f.read()
                self._template_cache[task_type] = template
                return template
                
        return None
```

### 2. Progress Indicators

Add progress indicators for long-running operations:

```python
def run_evolution(self, settings):
    # ...
    total_gens = settings["gens"]
    for gen in range(1, total_gens + 1):
        print_color(f"\nGeneration {gen}/{total_gens} [{gen*100//total_gens}%]", Colors.BOLD)
        # ...
```

### 3. Intermediate Results

Save intermediate results to allow resuming interrupted runs:

```python
def save_checkpoint(self, engine, generation, settings):
    """Save engine state to allow resuming."""
    checkpoint_file = self.session.base_dir / f"checkpoint_gen_{generation}.json"
    
    checkpoint_data = {
        "generation": generation,
        "settings": settings,
        "best_fitness": engine.get_best_solution().fitness,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
```

## Bluetooth Scanner Integration

When implementing the Bluetooth scanning functionality, pay special attention to:

1. **Device Discovery API**
   - Use the proper Bluetooth library functions
   - Handle device discovery timeouts
   - Process device metadata correctly

2. **Update Interval Implementation**
   - Ensure the update interval is respected
   - Implement non-blocking scanning where possible
   - Add a clean shutdown mechanism

3. **Output Formatting**
   - Implement table formatting with proper alignment
   - Add signal strength visualization (e.g., bars)
   - Sort devices by signal strength or discovery time

## Configuration Validation

Implement robust configuration validation:

```python
def validate_settings(self, settings):
    """Validate and normalize settings."""
    # Required fields
    required = ["task", "output_dir", "pop_size", "gens"]
    for field in required:
        if field not in settings:
            raise ValueError(f"Missing required setting: {field}")
    
    # Numeric ranges
    if settings["pop_size"] < 5:
        logging.warning(f"Population size too small, setting to minimum (5)")
        settings["pop_size"] = 5
    elif settings["pop_size"] > 200:
        logging.warning(f"Population size too large, setting to maximum (200)")
        settings["pop_size"] = 200
        
    # Probability ranges
    for prob_field in ["mutation_rate", "crossover_rate"]:
        if settings[prob_field] < 0:
            settings[prob_field] = 0
        elif settings[prob_field] > 1:
            settings[prob_field] = 1
            
    return settings
```

## Node Integration

To ensure proper integration between the different components of the system:

1. **Data Flow**
   - Document clear data interfaces between components
   - Use type hints to enforce interface contracts
   - Implement validation at component boundaries

2. **Component Lifecycle**
   - Initialize components in the correct order
   - Ensure proper cleanup of resources
   - Handle component failures gracefully

3. **Configuration Propagation**
   - Pass relevant configuration to each component
   - Use a consistent configuration schema
   - Validate configuration at each level

By following these implementation guidelines, you will create a robust, maintainable, and efficient rewrite of the progremon.py file that properly integrates with all components and handles the Bluetooth scanning functionality correctly.