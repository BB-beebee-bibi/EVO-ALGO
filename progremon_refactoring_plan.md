# Progremon Refactoring Plan

This document outlines the specific changes needed to transform the current `progremon.py` into a robust, maintainable implementation that properly handles Bluetooth scanning and other evolution tasks.

## Key Issues in Current Implementation

1. **AdaptiveTweaker Integration**
   - Calls non-existent `update_parameters` method instead of `record_metrics`
   - Missing compatibility with `Colors` and `print_color`
   - Parameter adjustments not applied to evolution engine

2. **Ethical Boundary Enforcement**
   - Cannot access code correctly from genome objects
   - Missing task-specific ethical boundaries
   - Not properly integrated in evaluation process

3. **Template Integration**
   - No proper template loading for task types
   - Missing initialization with templates
   - No task-specific guidance for Bluetooth scanning

4. **Error Handling**
   - Overly broad exception handling
   - Minimal logging
   - No recovery mechanisms

5. **File Management**
   - Basic output directory structure
   - Potential file overwriting
   - No session tracking

## Required Changes

### 1. Code Structure Changes

```diff
+ # Add new imports
+ import time
+ import datetime
+ import traceback
+ from pathlib import Path
+ 
+ # Add new classes
+ class TaskTemplateLoader:
+     """Loads and configures task-specific code templates."""
+     # ...
+ 
+ class EvolutionSession:
+     """Manages evolution session details and output files."""
+     # ...
+ 
  class ProgemonTrainer:
      def __init__(self):
-         self.adaptive_tweaker = AdaptiveTweaker(self.settings)
+         self.adaptive_tweaker = AdaptiveTweaker(self.settings)
+         self.template_loader = TaskTemplateLoader(self.settings["input_dir"])
+         self.session = None
+         self._setup_logging()
      
+     def _setup_logging(self):
+         """Set up enhanced logging."""
+         # ...
+     
+     def _configure_ethical_boundaries(self, enforcer, settings):
+         """Configure ethical boundaries based on task."""
+         # ...
```

### 2. Method Implementation Changes

```diff
  def parse_request(self, request: str) -> Dict[str, Any]:
-     config = self.settings.copy()
-     request_lower = request.lower()
-     
-     if "bluetooth" in request_lower:
-         config.update({
-             "task": "bluetooth_scan",
-             "pop_size": 50,
-             "gens": 30,
-             # ...
-         })
-     else:
-         config.update({
-             "task": "general",
-             # ...
-         })
+     config = self.settings.copy()
+     request_lower = request.lower()
+     
+     # Detect task type
+     if any(keyword in request_lower for keyword in ["bluetooth", "ble", "bt device"]):
+         task_type = "bluetooth_scan"
+     else:
+         task_type = "general"
+     
+     # Load task-specific configuration
+     task_config = self.template_loader.get_task_config(task_type)
+     config.update(task_config)
+     config["task"] = task_type
+     
+     # Extract update interval if specified
+     if "update" in request_lower and "second" in request_lower:
+         # Parse update interval
+         # ...
+     
+     # Extract output format if specified
+     if "table" in request_lower:
+         config["output_format"] = "table"
+     elif "json" in request_lower:
+         config["output_format"] = "json"
```

```diff
  def run_evolution(self, settings: Dict[str, Any]) -> bool:
      try:
-         # Set up output directory
-         output_dir = Path(settings["output_dir"])
-         output_dir.mkdir(parents=True, exist_ok=True)
+         # Create a new evolution session
+         self.session = EvolutionSession(
+             base_dir=settings["output_dir"],
+             task_type=settings["task"]
+         )
+         logging.info(f"Starting evolution session {self.session.session_id}")

          # Initialize evolution components
          evaluator = FitnessEvaluator()
          enforcer = EthicalBoundaryEnforcer()
-         
-         # Configure ethical boundaries
-         if settings["task"] == "bluetooth_scan":
-             enforcer.add_boundary("max_execution_time", max_execution_time=settings["max_execution_time"])
-             # ...
+         
+         # Configure ethical boundaries based on task type
+         self._configure_ethical_boundaries(enforcer, settings)

          # Initialize evolution engine
          engine = EvolutionEngine(
              # ...
          )
          
-         # Initialize population with task description
-         engine.initialize_population(
-             size=settings["pop_size"],
-             description=settings["description"],
-             task_type=settings["task"]
-         )
+         # Load task template if available
+         template_code = self.template_loader.load_template(settings["task"])
+         template_info = {"template_code": template_code} if template_code else {}
+         
+         # Initialize population with task description and template
+         engine.initialize_population(
+             size=settings["pop_size"],
+             task_description=settings["description"],
+             task_type=settings["task"],
+             **template_info
+         )

          # Run evolution
          for gen in range(1, settings["gens"] + 1):
              # ...
              
-             # Save current generation if requested
-             if settings["save_all_generations"]:
-                 gen_dir = output_dir / f"generation_{gen}"
-                 gen_dir.mkdir(exist_ok=True)
-                 
-                 # Save best solution
-                 best_solution = engine.get_best_solution()
-                 with open(gen_dir / "best.py", "w") as f:
-                     f.write(best_solution.to_source())
+             # Save current generation if requested
+             if settings["save_all_generations"]:
+                 gen_dir = self.session.get_generation_dir(gen)
+                 
+                 # Save best solution
+                 best_solution = engine.get_best_solution()
+                 with open(gen_dir / "best.py", "w") as f:
+                     f.write(best_solution.to_source())
+                 
+                 # Save generation metrics
+                 with open(gen_dir / "metrics.json", "w") as f:
+                     json.dump(population_metrics[-1], f, indent=2)
+             
+             # Update session stats
+             self.session.update_stats(gen, best_fitness)
              
-             # Apply adaptive tweaking
-             self.adaptive_tweaker.update_parameters(
-                 avg_fitness,
-                 best_fitness
-             )
+             # Apply adaptive tweaking using the proper method
+             self.adaptive_tweaker.record_metrics(
+                 population=population,
+                 best_fitness=best_fitness,
+                 avg_fitness=avg_fitness
+             )
+             
+             # Get adjusted parameters
+             new_params = self.adaptive_tweaker.adjust_parameters()
+             
+             # Apply parameter changes to the engine
+             if engine.mutation_rate != new_params["mutation_rate"]:
+                 print_color(
+                     f"Adjusting mutation rate: {engine.mutation_rate:.3f} -> {new_params['mutation_rate']:.3f}",
+                     Colors.YELLOW
+                 )
+                 engine.mutation_rate = new_params["mutation_rate"]
```

### 3. Add New Methods

```python
def _configure_ethical_boundaries(self, enforcer: EthicalBoundaryEnforcer, settings: Dict[str, Any]) -> None:
    """Configure ethical boundaries based on task type and settings."""
    # Common boundaries for all tasks
    enforcer.add_boundary("no_eval_exec")
    enforcer.add_boundary("no_destructive_operations")
    
    # Task-specific boundaries
    if settings["task"] == "bluetooth_scan":
        enforcer.add_boundary(
            "max_execution_time", 
            max_execution_time=settings.get("max_execution_time", 10.0)
        )
        enforcer.add_boundary(
            "max_memory_usage", 
            max_memory_usage=settings.get("max_memory_usage", 500)
        )
        enforcer.add_boundary(
            "allowed_imports", 
            allowed_imports=settings.get("allowed_libraries", ["bluetooth"])
        )
        # Add bluetooth-specific boundaries
        enforcer.add_boundary(
            "no_continuous_scanning", 
            max_scan_time=settings.get("max_scan_time", 30.0)
        )
```

```python
def _load_bluetooth_template(self) -> str:
    """Load the Bluetooth scanner template with correct imports."""
    template = self.template_loader.load_template("bluetooth_scan")
    if not template:
        # Fallback template if none found
        template = """
import bluetooth
import time
from typing import List, Dict, Any

def scan_bluetooth_devices() -> List[Dict[str, Any]]:
    # ... Template implementation ...
"""
    return template
```

### 4. Modify AdaptiveTweaker

Changes needed in `adaptive_tweaker.py`:

```diff
+ # Import color utilities from progremon
+ from progremon import Colors, print_color

  class AdaptiveTweaker:
      # ... existing code ...
      
+     def update_parameters(self, avg_fitness: float, best_fitness: float) -> None:
+         """
+         Legacy method for compatibility with the original Progremon.
+         This method exists as a bridge to the record_metrics method.
+         """
+         # Create a mock population with the provided fitness values
+         mock_population = [
+             type('MockGenome', (), {'fitness': avg_fitness}),
+             type('MockGenome', (), {'fitness': best_fitness})
+         ]
+         
+         # Call the actual method
+         self.record_metrics(mock_population, best_fitness, avg_fitness)
```

## Expected Benefits

### 1. Improved Reliability

- **Before:** System fails when encountering errors in genome evaluation or parameter tweaking
- **After:** Robust error handling allows the system to continue even when parts fail

### 2. Better Code Organization

- **Before:** Monolithic implementation with mixed responsibilities
- **After:** Clear separation of concerns with dedicated classes for:
  - Task template management
  - Session tracking
  - Ethical boundary configuration

### 3. Enhanced Adaptability

- **Before:** Limited adaptation to different tasks and evolving fitness landscapes
- **After:** Task-specific configurations and proper adaptive parameter tweaking

### 4. Better Output Management

- **Before:** Basic file output with potential overwriting
- **After:** Structured session directories with:
  - Metadata tracking
  - Per-generation metrics
  - Unique session identifiers

### 5. Improved Bluetooth Scanner Evolution

- **Before:** Generic evolution that doesn't leverage Bluetooth-specific knowledge
- **After:** Task-specific:
  - Parameter configuration
  - Ethical boundaries
  - Template integration
  - Output formatting

## Implementation Priority

1. **Core Structure** (High Priority)
   - Add new classes (TaskTemplateLoader, EvolutionSession)
   - Set up improved logging

2. **Fix Integration Issues** (High Priority)
   - Fix AdaptiveTweaker integration
   - Fix EthicalBoundaryEnforcer integration
   - Fix template integration

3. **Enhance Error Handling** (Medium Priority)
   - Add better exception handling
   - Implement recovery mechanisms
   - Add detailed logging

4. **Improve Output Management** (Medium Priority)
   - Implement session-based directories
   - Add metadata tracking
   - Implement unique file naming

5. **Optimize Evolution Process** (Lower Priority)
   - Add progress indicators
   - Implement checkpointing
   - Add performance optimizations

## Testing Approach

For each implemented change, test in isolation and then in integration:

1. **Unit Tests**
   - Test TaskTemplateLoader with various template types
   - Test EvolutionSession file management
   - Test ethical boundary configuration

2. **Integration Tests**
   - Test ProgemonTrainer with AdaptiveTweaker
   - Test ProgemonTrainer with EthicalBoundaryEnforcer
   - Test full evolution process

3. **End-to-End Test**
   - Run complete evolution with Bluetooth scanner request
   - Verify output files and solution quality

By following this refactoring plan, the progremon.py implementation will be transformed into a robust, maintainable system that properly integrates with all components and successfully evolves Bluetooth scanning functionality.