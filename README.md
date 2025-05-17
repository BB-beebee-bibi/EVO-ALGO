# TRISOLARIS Evolutionary Code Engine

A next-generation evolutionary computation framework for evolving Python code through adaptive, multi-objective optimization.

## Architecture Overview

TRISOLARIS implements a layered ecosystem for code evolution:

1. **Meta-Evolutionary Control Layer**
   - Adaptive parameter tuning
   - Diversity monitoring with Shannon entropy
   - Resource allocation management
   - Performance-based feedback loops

2. **Dual Population Management**
   - Co-evolution of programs and operators
   - Operator success rate tracking
   - Guided statistical sampling
   - Genetic variance optimization

3. **Graduated Validation Pipeline**
   - Syntax validation
   - Static semantic analysis
   - Lightweight execution sampling
   - Comprehensive test suite execution

4. **Safety and Governance Layer**
   - Resource monitoring
   - Plagiarism detection
   - Ethical alignment validation
   - Audit trail generation

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run type checking
mypy trisolaris/

# Run linting
pylint trisolaris/
```

## Project Structure

```
trisolaris/
├── core/                 # Core evolution engine
│   ├── meta_control/    # Meta-evolutionary control
│   ├── population/      # Population management
│   └── operators/       # Evolution operators
├── evolution/           # Evolution strategies
│   ├── programs/        # Program evolution
│   └── operators/       # Operator evolution
├── validation/          # Validation pipeline
│   ├── syntax/         # Syntax validation
│   ├── semantic/       # Semantic analysis
│   ├── execution/      # Execution testing
│   └── property/       # Property verification
├── governance/          # Safety and governance
│   ├── monitoring/     # Resource monitoring
│   ├── plagiarism/     # Plagiarism detection
│   └── ethics/         # Ethical validation
└── utils/              # Utility functions

tests/                  # Test suite
├── unit/              # Unit tests
├── integration/       # Integration tests
└── benchmarks/        # Performance benchmarks

docs/                  # Documentation
├── architecture/      # Architecture docs
├── api/              # API documentation
└── examples/         # Usage examples
```

## Key Features

- **Adaptive Evolution**: Parameters automatically adjust based on population metrics
- **Dual Population**: Programs and operators co-evolve for optimal results
- **Graduated Validation**: Multi-stage validation with confidence scoring
- **Safety First**: Built-in resource monitoring and ethical validation
- **Empirical Focus**: All components require measurable improvement

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details
