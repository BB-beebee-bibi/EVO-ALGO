# 🚀 Progremon: Enhanced Evolution Engine

> "Gotta evolve 'em all!"

Progremon is an evolutionary code generation platform that uses the Trisolaris evolution engine to evolve program solutions based on user requirements. This enhanced version addresses integration issues, improves error handling, enhances ethical boundary enforcement, and adds specialized task functionality.

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [Task Types](#task-types)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## 🏗️ System Architecture

The enhanced Progremon system features a modular architecture with clear component boundaries and improved integration:

```
                 ┌─────────────────┐
                 │  ProgemonTrainer│
                 │  (Main Interface)│
                 └────────┬────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
┌─────────▼───────────┐        ┌─────────▼────────┐
│  TaskTemplateLoader │        │  EvolutionSession│
│  (Template Management)│        │  (Session Management)│
└─────────────────────┘        └─────────┬────────┘
                                         │
                               ┌─────────▼────────┐
                               │  EvolutionEngine │
                               │  (Trisolaris Core)│
                               └─────────┬────────┘
                                         │
                        ┌────────────────┼───────────────┐
                        │                │               │
               ┌────────▼─────┐  ┌───────▼──────┐ ┌──────▼───────┐
               │ CodeGenome   │  │FitnessEvaluator│ │AdaptiveTweaker│
               │(Solution Repr)│  │(Fitness Calc) │ │(Param Adjust) │
               └──────────────┘  └──────┬────────┘ └──────────────┘
                                        │
                                ┌───────▼───────┐
                                │EthicalBoundary│
                                │  Enforcer    │
                                └───────────────┘
```

## 🧩 Key Components

The system consists of the following key components:

### ProgemonTrainer
Main interface that handles user interaction, request parsing, and orchestrates the evolution process.

### TaskTemplateLoader
Manages task-specific templates with caching for better performance.

### EvolutionSession
Handles session management, including unique session ID generation, directory structure, and metadata tracking.

### BluetoothScanner
Specialized component for Bluetooth device discovery with error handling and configurable parameters.

### Utilities
Shared functionality for color printing, logging, file operations, and helper functions.

## 📥 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/username/progremon.git
cd progremon
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python test_progremon_enhanced.py
```

## 🚀 Usage

The enhanced Progremon system can be used in either interactive or command-line mode:

### Interactive Mode

```bash
python progremon_enhanced.py
```

This will start the interactive interface where you can:
1. Describe the program you want to evolve
2. Customize evolution settings
3. Watch the evolution process
4. Get your evolved solution

### Command-line Mode

```bash
python progremon_enhanced.py --task bluetooth_scan --description "Scan for nearby Bluetooth devices" --output-dir evolved_bluetooth
```

Available options:
- `--task`: Specify the task type (e.g., bluetooth_scan, general)
- `--description`: Provide a description of what you want to evolve
- `--output-dir`: Set the output directory for evolved code

## 🛠️ Task Types

The enhanced Progremon system supports specialized task types:

### General Purpose (Default)
General-purpose code evolution with standard settings.

### Bluetooth Scanning
Specialized template and settings for Bluetooth device discovery:
- Proper device discovery with error handling
- Support for table, JSON, or text output formats
- Configurable scan parameters
- Privacy-respecting scanning behavior
- Cross-platform support (Linux, Windows, macOS)

## 🔧 Advanced Features

### Session Management
The system now supports session management with:
- Unique session IDs
- Directory structure management
- Metadata tracking
- Session recovery for interrupted evolutions

### Ethical Boundary Enforcement
Enhanced ethical boundaries that:
- Scale with solution quality
- Include task-specific configurations
- Monitor resource usage
- Ensure privacy-respecting behavior

### Adaptive Parameter Tuning
The AdaptiveTweaker component automatically adjusts evolution parameters based on:
- Population diversity
- Fitness improvement rate
- Success/failure patterns
- Task-specific heuristics

## 🔍 Troubleshooting

### Common Issues

**Issue**: Error importing Bluetooth libraries
**Solution**: Install the appropriate libraries for your platform:
- Linux: `pip install pybluez bleak`
- Windows: `pip install bleak`
- macOS: `pip install bleak`

**Issue**: Evolution process seems to get stuck
**Solution**: Check if there are permission issues with file writing or if the system is running out of memory. Try reducing the population size.

**Issue**: No valid solutions evolved
**Solution**: Try increasing the number of generations or adjusting mutation/crossover rates. Also, check if templates are correctly configured for your task.

### Logs

Logs are stored in the session directory and can help diagnose issues:
- `session.log`: Contains detailed information about the evolution process
- `session_info.json`: Contains metadata about the session

## 💻 Development

### Running Tests

```bash
python test_progremon_enhanced.py
```

### Project Structure

```
progremon/
├── progremon_enhanced.py      # Main interface
├── utils.py                   # Shared utilities
├── task_template_loader.py    # Template management
├── evolution_session.py       # Session management
├── bluetooth_scanner.py       # Bluetooth functionality
├── test_progremon_enhanced.py # Test suite
├── guidance/                  # Task templates
│   └── bluetooth_scanner_template.py
└── evolved_output/            # Generated solutions
```

### Adding a New Task Type

1. Create a template file in the `guidance` directory
2. Update `TaskTemplateLoader.templates` to include your new task type
3. Add task-specific configuration in `ProgemonTrainer.parse_request`
4. Add task-specific ethical boundaries in `_configure_ethical_boundaries`

## 📊 Evolution Metrics

The system collects and reports metrics throughout the evolution process:
- Average fitness per generation
- Best fitness per generation
- Parameter adjustment history
- Ethical boundary violations

These metrics can be found in the session directory after evolution completes.

## 🙏 Acknowledgments

- The [Trisolaris](https://github.com/username/trisolaris) evolution engine for providing the core evolution functionality
- Contributors to the original Progremon system
- The open-source community for various libraries used in this project

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.