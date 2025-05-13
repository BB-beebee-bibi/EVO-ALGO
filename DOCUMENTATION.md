# Progrémon Project Documentation Hub

## Project Overview

The Progrémon project is an advanced evolutionary computing framework designed to automatically generate specialized programs (Progrémons) through guided evolution. At its core is the Trisolaris evolutionary engine, a sophisticated system that applies natural selection principles to iteratively improve code, configurations, and designs.

Originally developed under the directory name EVO-ALGO, the project has evolved into a modular, task-based architecture that separates the evolutionary engine (Trisolaris) from the specific tasks it can evolve solutions for. This separation allows for greater flexibility and extensibility, enabling the framework to tackle a wide range of programming challenges.

The Progrémon project represents a significant step toward autonomous program evolution guided by clearly defined constraints and objectives. It combines cutting-edge evolutionary algorithms with strict ethical boundaries and resource-aware execution to create practical, efficient, and safe software solutions.

Recent architectural enhancements have integrated rigorous mathematical foundations from evolutionary theory, implemented a secure sandboxed evolution environment, and developed a comprehensive post-evolution ethical evaluation system. These components work together to create a robust framework that bridges theoretical evolutionary principles with practical implementation.

## Component Architecture

### Trisolaris Evolutionary Engine

Trisolaris is the core evolutionary computation framework that powers the Progrémon project. Named as a reference to the science fiction novel "The Three-Body Problem" by Liu Cixin, it implements sophisticated evolutionary algorithms to generate, evaluate, and refine code solutions.

The Trisolaris engine consists of several key components that work together in a bidirectional manner:

1. **Mathematical Foundation**: Implements rigorous evolutionary theory principles including:
   - **Price Equation**: Decomposes evolutionary change into selection and transmission components
   - **Fisher's Fundamental Theorem**: Predicts the rate of fitness increase based on genetic variance
   - **Selection Gradients**: Guides evolution toward promising regions of the solution space

2. **Adaptive Landscape Navigator**: Models fitness landscapes for code evolution, allowing visualization and efficient traversal of the solution space. Uses mathematical models from evolutionary theory to guide the search process and adapt to changing fitness landscapes.

3. **Sandboxed Evolution Environment**: Provides a secure, isolated environment for code evolution with:
   - Resource monitoring and constraints (CPU, memory, execution time)
   - File system isolation and simulated access
   - Network isolation and simulated connections
   - Process isolation and security boundaries

4. **Post-Evolution Ethical Evaluation**: Implements a multi-layered approach to ensure evolved code meets ethical standards:
   - Syntax checking and functionality verification
   - Ethical assessment against principles of privacy, security, fairness, transparency, and accountability
   - Gurbani alignment evaluation

5. **Genome Repository**: Provides versioned storage of code genomes using Git integration, with phylogenetic tracking of solution lineages. Maintains archives of both successful and failed variants.

6. **Island Ecosystem Manager**: Maintains multiple subpopulations with different selection pressures and enables cross-pollination between islands.

7. **Resource Steward**: Monitors system resources and maintains ≥25% availability, dynamically adjusting evolution pace based on resource availability.

8. **Diversity Guardian**: Tracks population metrics and implements strategies to maintain genetic diversity.

9. **Evolution Monitor & Visualizer**: Provides tools for tracking evolution progress and visualizing results.

10. **Task Interface**: Defines a generic interface for evolvable tasks.

#### Component Interactions

The Trisolaris engine components interact bidirectionally, creating a sophisticated feedback system:

1. **Mathematical Foundation ↔ Evolution Engine**
   - Mathematical models guide selection pressure and fitness evaluation
   - Evolution results feed back to refine mathematical models
   - Price equation decomposes evolutionary change to inform next generations
   - Fisher's theorem predicts rate of fitness increase to optimize evolution parameters

2. **Sandboxed Environment ↔ Evolution Engine**
   - Sandbox provides safe execution environment for candidate solutions
   - Resource usage data feeds back to evolution engine to optimize resource efficiency
   - Evolution engine adapts to sandbox constraints

3. **Ethical Evaluation ↔ Evolution Engine**
   - Post-evolution ethical assessment filters solutions
   - Ethical evaluation results feed back to guide future evolution
   - Multi-layered approach ensures solutions meet all requirements

### Progrémon Programs

Progrémons are the specialized programs generated by the Trisolaris evolutionary engine. Each Progrémon is evolved to perform a specific task, optimized through generations of selection and refinement. 

The Progrémon interface serves as the primary way users interact with the system. Users input requirements and constraints through this interface, which then translates these inputs into mathematical selection environments that guide the evolution process. This translation process is a key innovation that allows non-technical users to harness the power of evolutionary computation.

For example, when a user requests "a network scanner that identifies IoT devices but doesn't attempt to access them," the Progrémon interface translates this into specific selection pressures and fitness landscape parameters that guide the evolution toward solutions meeting these requirements.

The current implementation includes several types of Progrémons:

1. **Drive Scanner Progrémon**: Scans connected storage devices, creates detailed snapshots of drive contents, and identifies file types and organizes information.

2. **Network Scanner Progrémon**: Scans local networks for connected devices, identifies device types including IoT devices, and reports detailed information about discovered devices.

3. **Bluetooth Scanner Progrémon**: Scans for Bluetooth devices with a focus on security vulnerability detection.

## Theoretical Foundations

The Trisolaris framework implements key concepts from evolutionary theory, now with a rigorous mathematical foundation:

### Price Equation Implementation

The Price equation is a fundamental mathematical theorem that describes how trait values change from one generation to the next. It decomposes evolutionary change into two components:

1. **Selection Component**: Covariance between fitness and trait value
2. **Transmission Component**: Expected value of fitness times change in trait value

Mathematically expressed as: ΔZ = Cov(w, z)/w̄ + E(w·Δz)/w̄

Where:
- ΔZ is the change in the average value of trait z
- w is fitness
- z is the trait value
- w̄ is the average fitness
- Cov(w, z) is the covariance between fitness and trait value
- E(w·Δz) is the expected value of the product of fitness and the change in trait value

**Technical Implementation**: The Trisolaris engine uses the Price equation to analyze how code traits (such as function length, complexity, or specific patterns) change across generations. This allows the system to understand which traits are being selected for and which are being selected against, providing insights into the evolutionary dynamics.

**Practical Example**: When evolving a network scanner, the Price equation helps decompose the evolutionary change into:
- Selection for code patterns that efficiently identify devices
- Selection against code patterns that attempt to access devices
- Transmission of beneficial traits to the next generation

**Non-Technical Explanation**: Think of the Price equation like a recipe analyzer that tells you which ingredients in your dish are making people like it more (selection component) and how those ingredients change when you adjust the recipe slightly (transmission component). This helps the system understand which parts of the code are working well and should be kept, and which parts need to be changed.

### Fisher's Fundamental Theorem

Fisher's Fundamental Theorem of Natural Selection states that the rate of increase in fitness of a population at any time is equal to the genetic variance in fitness at that time.

Mathematically expressed as: ΔW = VA/W

Where:
- ΔW is the rate of increase in fitness
- VA is the additive genetic variance in fitness
- W is the mean fitness

**Technical Implementation**: The Trisolaris engine uses Fisher's theorem to predict how quickly the fitness of the population will increase, which helps optimize evolution parameters such as population size, mutation rate, and selection pressure.

**Practical Example**: When evolving a drive scanner that preserves user privacy, Fisher's theorem helps predict:
- Rate of improvement for snapshot functionality
- Optimal balance between thoroughness and privacy preservation
- Expected number of generations needed to reach a target fitness level

**Non-Technical Explanation**: Fisher's theorem is like a progress predictor that tells you how quickly your solutions will improve based on how much variety you have in your current set of solutions. More variety (genetic variance) means faster improvement, which helps the system decide how many different approaches to try in parallel.

### Additional Evolutionary Concepts

1. **Fitness Landscapes and Adaptive Walks**: Mathematical models that represent the relationship between genotypes (code structures) and their fitness (performance). The framework navigates these landscapes to find optimal solutions.

2. **Selection Gradients and Differential Reproduction**: Mechanisms that favor the reproduction of higher-fitness solutions, gradually improving the population over generations.

3. **Mutation-Selection Balance**: The equilibrium between introducing variation through mutations and removing harmful variations through selection.

4. **Exploration-Exploitation Trade-offs**: Balancing the search for new, potentially better solutions (exploration) with refining existing good solutions (exploitation).

5. **Population Genetics Principles**: Concepts from biological evolution applied to code evolution, including genetic drift, gene flow between islands, and selection pressure.

### Framework Priorities

The framework operates on three fundamental priorities:

1. **Alignment (60%)**: Solutions must adhere to universal principles including:
   - Service to others rather than self-interest
   - Truthful and transparent design
   - Resource harmony and mindfulness
   - Inclusive and respectful language
   - Humble, simple approaches over complexity

2. **Functionality (25%)**: Code must work correctly, satisfying all requirements and test cases.

3. **Efficiency (15%)**: Solutions should minimize resource usage, execution time, and complexity.

## Ethical Boundaries

The Trisolaris framework has evolved from pre-evolution ethical constraints to a comprehensive post-evolution ethical evaluation system. This multi-layered approach ensures that evolved code meets high ethical standards while still allowing for creative evolution.

### Post-Evolution Ethical Evaluation System

The post-evolution ethical evaluation system consists of three main layers:

1. **Syntax Checking**
   - Verifies that the evolved code is syntactically valid
   - Ensures the code can be parsed and executed
   - Prevents runtime errors and exceptions

2. **Functionality Verification**
   - Confirms that the code performs its intended function
   - Validates against test cases and requirements
   - Ensures the solution actually solves the given problem

3. **Ethical Assessment**
   - Evaluates code against ethical principles:
     - **Privacy**: Respects user data and prevents unauthorized access
     - **Security**: Follows secure coding practices and prevents vulnerabilities
     - **Fairness**: Treats all users equitably and avoids bias
     - **Transparency**: Makes operations clear and understandable
     - **Accountability**: Ensures actions can be traced and explained
   - Checks alignment with Gurbani principles:
     - Service to others rather than self-interest
     - Truthful and transparent design
     - Resource harmony and mindfulness
     - Inclusive and respectful language

### Practical Examples

**Example 1: Network Scanner**

When evaluating a network scanner Progrémon, the ethical evaluation system:
- Verifies the code can correctly identify devices (functionality)
- Ensures it doesn't attempt to access devices without permission (privacy)
- Confirms it doesn't store or transmit sensitive information (security)
- Checks that it treats all device types equally (fairness)
- Validates that it clearly reports what it's doing (transparency)

**Technical Explanation**: The system uses static analysis to identify code patterns that might violate privacy, such as attempts to access device data beyond basic identification. It also checks for secure coding practices like input validation and proper error handling.

**Non-Technical Explanation**: Think of the ethical evaluation like a security checkpoint that ensures the evolved program follows the rules. It checks that the program only looks at what it's allowed to look at, doesn't try to access private information, and is honest about what it's doing.

### Feedback Loop to Evolution

A key innovation in the post-evolution ethical evaluation system is the feedback loop to the evolution process. When ethical issues are identified:

1. The specific concerns are documented and categorized
2. This information feeds back into the evolution process
3. Future generations are guided away from problematic patterns
4. The system learns over time which approaches are ethically sound

This creates a self-improving system that becomes increasingly adept at generating ethical solutions.

## Task System

The Progrémon project implements a task-based architecture that separates the evolution process from specific tasks:

1. **Task Interface**: Defines a common interface that all evolvable tasks must implement, including:
   - `get_name()`: Return the task name
   - `get_description()`: Return a description of the task
   - `get_template()`: Return template code to start from
   - `evaluate_fitness()`: Evaluate the fitness of a solution

2. **Task Implementations**: Task-specific code that handles fitness evaluation, templates, and evolution parameters.

3. **Task Runner**: Generic evolution runner that can evolve any task that implements the TaskInterface.

This separation allows for:
- Reusing the same evolutionary engine for different tasks
- Defining task-specific fitness functions and templates
- Applying task-specific post-processing to evolved solutions
- Evolving new tasks without modifying the core engine

## Debug and Testing Infrastructure

The Progrémon project includes comprehensive debug utilities to help with debugging, performance monitoring, and troubleshooting:

### Debug Components

1. **Debug Logging Module**: A comprehensive debug logging module that provides multi-level logging, function call tracing with performance metrics, genome content logging, fitness evaluation details, ethical boundary check results, resource usage monitoring, evolution progress tracking, and thread-safe logging for concurrent operations.

2. **Debug Task Runner**: An enhanced version of the standard task runner with comprehensive debug logging, performance monitoring and reporting, detailed progress tracking, exception handling and reporting, and resource usage monitoring.

3. **Task-Specific Debug Scripts**: Convenience scripts for running specific tasks with debug capabilities.

### Testing Infrastructure

The project includes a testing framework with:

1. **Unit Tests**: Tests for individual components of the Trisolaris engine.
2. **Integration Tests**: Tests for the interaction between components.
3. **Functional Tests**: Tests for complete task evolution processes.

## Getting Started Guide

### Running a Task Evolution

To evolve a solution for a specific task, use the task runner:

```bash
python3 trisolaris/task_runner.py drive_scanner --template drive_scanner.py --pop-size 20 --gens 10 --ethics-level full --resource-monitoring
```

### Command-line Options

- `task`: Name of the task to evolve (e.g., drive_scanner)
- `--template`: Path to a custom template file (optional)
- `--output-dir`: Base directory to save evolved code (default: outputs)
- `--pop-size`: Population size (default: task-specific recommendation)
- `--gens`: Number of generations (default: task-specific recommendation)
- `--mutation-rate`: Mutation rate (default: task-specific recommendation)
- `--crossover-rate`: Crossover rate (default: task-specific recommendation)
- `--ethics-level`: Ethical filter level (none, basic, full) (default: basic)
- `--resource-monitoring`: Enable resource monitoring and throttling
- `--use-git`: Use Git for version control of solution history
- `--use-islands`: Use island model for evolution
- `--islands`: Number of islands when using island model (default: 3)
- `--migration-interval`: Number of generations between migrations (default: 3)
- `--diversity-threshold`: Diversity threshold for injection (default: 0.3)

### Creating a New Task

To create a new task for the Trisolaris engine:

1. Create a new class that implements the `TaskInterface` in `trisolaris/tasks/`
2. Implement all required methods
3. Register the task in `TASK_REGISTRY` in `trisolaris/task_runner.py`

### Running with Debug Capabilities

For debugging and performance monitoring, use one of the task-specific debug scripts:

```bash
# Run network scanner evolution with debug capabilities
./debug_network_scanner.py --gens=5 --pop-size=20 --debug-level=verbose

# Run drive scanner evolution with resource monitoring
./debug_drive_scanner.py --resource-monitoring --debug-level=trace

# Run bluetooth scanner evolution with island model
./debug_bluetooth_scanner.py --use-islands --islands=3 --debug-level=verbose
```

## System Architecture Diagram

```
                                 TRISOLARIS FRAMEWORK
                                 ===================
                                         |
                +--------------------+    |    +----------------------+
                |                    |<-->|<-->|                      |
                | MATHEMATICAL       |    |    | SANDBOXED            |
                | FOUNDATION         |    |    | ENVIRONMENT          |
                |                    |    |    |                      |
                | - Price Equation   |    |    | - Resource Limits    |
                | - Fisher's Theorem |    |    | - File System Access |
                | - Selection        |    |    | - Network Simulation |
                |   Gradients        |    |    | - Process Isolation  |
                |                    |    |    |                      |
                +--------+-----------+    |    +-----------+----------+
                         ^                |                ^
                         |                |                |
                         v                v                v
                +--------------------------------------------------+
                |                                                  |
                |               EVOLUTION ENGINE                   |
                |                                                  |
                | +----------------+  +------------------------+   |
                | | Population     |  | Fitness Evaluation     |   |
                | | Management     |  | & Selection            |   |
                | +----------------+  +------------------------+   |
                |                                                  |
                +----------------------+---------------------------+
                                       ^
                                       |
                                       v
                +--------------------------------------------------+
                |                                                  |
                |         POST-EVOLUTION ETHICAL EVALUATION        |
                |                                                  |
                | +----------------+  +------------------------+   |
                | | Syntax &       |  | Ethical Assessment     |   |
                | | Functionality  |  | - Privacy, Security    |   |
                | | Verification   |  | - Fairness, Transparency|  |
                | |                |  | - Accountability       |   |
                | |                |  | - Gurbani Alignment    |   |
                | +----------------+  +------------------------+   |
                |                                                  |
                +----------------------+---------------------------+
                                       ^
                                       |
                                       v
                                 +-----------+
                                 | Progrémon |
                                 +-----------+
                                       ^
                                       |
                                       v
                                 +-----------+
                                 |   User    |
                                 +-----------+
```

## Evolution Workflow Diagram

```
+-------------+     +-------------------+     +----------------------+
|             |     |                   |     |                      |
|    User     +---->+    Progrémon      +---->+  Task Definition    |
|             |     |    Interface      |     |                      |
+-------------+     +-------------------+     +----------+-----------+
                                                         |
                                                         v
+-------------------+     +-------------------+     +----+-------------+
|                   |     |                   |     |                  |
| Fitness Landscape <-----+ Mathematical      <-----+ Selection        |
| Visualization     |     | Foundation        |     | Criteria         |
|                   |     |                   |     |                  |
+--------+----------+     +---+---------------+     +------------------+
         |                    ^
         v                    |
+--------+----------+     +---+---------------+     +------------------+
|                   |     |                   |     |                  |
| Initial Population+---->+ Evolution Engine  +---->+ Sandbox          |
| Generation        |     |                   |     | Environment      |
|                   |     |                   |     |                  |
+-------------------+     +---+---------------+     +--------+---------+
                              ^                              |
                              |                              v
+-------------------+     +---+---------------+     +--------+---------+
|                   |     |                   |     |                  |
| Diversity         +---->+ Population        <-----+ Resource         |
| Management        |     | Management        |     | Monitoring       |
|                   |     |                   |     |                  |
+-------------------+     +---+---------------+     +------------------+
                              |
                              v
+-------------------+     +---+---------------+     +------------------+
|                   |     |                   |     |                  |
| Fitness           <-----+ Candidate         +---->+ Syntax           |
| Evaluation        |     | Solutions         |     | Checking         |
|                   |     |                   |     |                  |
+-------------------+     +---+---------------+     +--------+---------+
                              |                              |
                              v                              v
+-------------------+     +---+---------------+     +--------+---------+
|                   |     |                   |     |                  |
| Ethical           <-----+ Post-Evolution    <-----+ Functionality    |
| Assessment        |     | Evaluation        |     | Verification     |
|                   |     |                   |     |                  |
+-------------------+     +---+---------------+     +------------------+
                              |
                              v
+-------------------+     +---+---------------+
|                   |     |                   |
| Feedback Loop     +---->+ Final Progrémon   |
| to Evolution      |     | Solution          |
|                   |     |                   |
+-------------------+     +-------------------+
```

## Glossary

| Term | Definition |
|------|------------|
| **Progrémon** | A specialized program generated by the Trisolaris evolutionary engine, evolved to perform a specific task. Also refers to the user-facing interface for inputting requirements. |
| **Trisolaris** | The evolutionary computation framework that powers the Progrémon project, named after the science fiction novel "The Three-Body Problem". |
| **Price Equation** | A mathematical theorem that decomposes evolutionary change into selection and transmission components. |
| **Fisher's Theorem** | A fundamental theorem stating that the rate of increase in fitness equals the genetic variance in fitness. |
| **Adaptive Landscape** | A mathematical model that represents the relationship between code structures and their fitness or performance. |
| **Code Genome** | A representation of code as a manipulable data structure (AST or graph) that can undergo mutation and crossover operations. |
| **Sandboxed Environment** | A secure, isolated execution environment that prevents harmful operations while allowing safe evolution. |
| **Post-Evolution Ethical Evaluation** | A multi-layered system that assesses evolved code for syntax correctness, functionality, and ethical compliance. |
| **Ethical Boundary Enforcer** | A component that implements hard constraints that all solutions must satisfy before they can be evaluated for fitness. |
| **Fitness Evaluation** | The process of assessing how well a solution performs against defined criteria. |
| **Island Model** | An evolutionary approach that maintains multiple subpopulations with different selection pressures. |
| **Mutation** | A random change to a code genome that introduces variation. |
| **Crossover** | The process of combining parts of two parent solutions to create a new child solution. |
| **Selection Pressure** | The degree to which better-performing solutions are favored for reproduction. |
| **Task Interface** | A common interface that all evolvable tasks must implement. |
| **Resource Steward** | A component that monitors system resources and adjusts the evolution process accordingly. |
| **Diversity Guardian** | A component that tracks population metrics and implements strategies to maintain genetic diversity. |
| **Evolution Monitor** | A tool for tracking evolution progress and visualizing results. |
| **Selection Gradient** | The direction and magnitude of selection acting on different traits. |
| **Gurbani Alignment** | Evaluation of solutions against ethical principles derived from Gurbani teachings. |