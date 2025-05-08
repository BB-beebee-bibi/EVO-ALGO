# Progremon: Evolutionary Code Generation Framework

Progremon is a framework for evolutionary code generation that leverages the Trisolaris evolution system to evolve functional code for various tasks, with special support for Bluetooth device scanning.

## Overview

Progremon provides a user-friendly interface to the Trisolaris evolutionary engine, allowing users to:

- Evolve code for specific tasks using natural language descriptions
- Customize evolution parameters 
- Track evolution progress
- Save and test evolved solutions

The system is designed to be robust, with comprehensive error handling, ethical boundaries enforcement, and adaptive parameter tuning.

## Features

- **Task-specific Templates**: Pre-configured templates for common tasks like Bluetooth scanning
- **Adaptive Parameter Tweaking**: Automatically adjusts mutation rates based on fitness progress
- **Ethical Boundary Enforcement**: Ensures evolved code follows ethical guidelines
- **Session-based Output Management**: Organizes evolved solutions in a structured directory system
- **Comprehensive Error Handling**: Robust error recovery mechanisms

## Getting Started

### Prerequisites

- Python 3.7+
- Trisolaris evolution engine
- Required Python packages: `numpy`

For Bluetooth scanning functionality:
- `bluetooth` Python package
- BlueZ (Linux) or equivalent Bluetooth library for your OS

### Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install numpy bluetooth
   ```

### Usage

#### Interactive Mode

```bash
python progremon_final.py
```

Follow the prompts to describe the task you want to evolve code for.

#### Command Line Mode

```bash
python progremon_final.py --task bluetooth_scan --desc "Find all nearby Bluetooth devices and display their signal strength" --pop 20 --gens 10
```

### Testing

Run the test script to verify proper functioning:

```bash
python test_progremon_final.py
```

## Core Components

1. **ProgemonTrainer**: Main class handling the evolution process
2. **TaskTemplateLoader**: Loads task-specific templates
3. **EvolutionSession**: Manages evolution sessions and output organization
4. **AdaptiveTweaker**: Adjusts evolution parameters based on performance

## Evolution Process

1. User provides a task description
2. System detects task type and configures appropriate settings
3. Evolution engine initializes population with task-specific templates
4. Generations are evolved with fitness evaluation and ethical boundary checks
5. Adaptive parameter tweaking is applied based on evolution progress
6. Best solution is saved and can be executed

## File Structure

- `progremon_final.py`: Main implementation
- `adaptive_tweaker_fix.py`: Fixed implementation of adaptive parameter tweaking
- `guidance/`: Directory containing task templates
  - `bluetooth_scanner_template.py`: Template for Bluetooth scanning
- `evolved_output/`: Default output directory for evolved code

## Notes

This implementation focuses on robust error handling, proper fitness evaluation, ethical boundary enforcement, and adaptive parameter tweaking. It is designed to handle both general code evolution tasks and specialized Bluetooth scanning functionality.