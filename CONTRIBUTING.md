# Contributing to EV Energy Prediction

First of all, thank you for considering contributing to the EV Energy Prediction project! This document provides guidelines and instructions for contributing to make the process smooth and effective for everyone involved.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment](#development-environment)
  - [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Guidelines](#commit-guidelines)
  - [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
  - [Python Style Guide](#python-style-guide)
  - [Documentation](#documentation)
  - [Testing](#testing)
- [Contribution Types](#contribution-types)
  - [Bug Reports](#bug-reports)
  - [Feature Requests](#feature-requests)
  - [Code Contributions](#code-contributions)
  - [Documentation Improvements](#documentation-improvements)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that establishes how we collaborate and interact with each other. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Development Environment

To set up the development environment:

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/ev-energy-prediction.git
   cd ev-energy-prediction
   ```

3. Set up the development environment:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. Verify the setup:
   ```bash
   # Run tests to ensure everything is working
   pytest
   ```

### Project Structure

Familiarize yourself with the project structure:

```
├── config/                  # Configuration files
│   └── model_config.yaml    # Model hyperparameters and settings
├── data/                    # Data storage and processing
│   └── preprocess.py        # Data preprocessing scripts
├── models/                  # Model implementations
│   ├── ensemble.py          # Ensemble model architecture
│   ├── lstm_model.py        # LSTM implementation
│   ├── train_model.py       # Model training utilities
│   └── xgboost_model.py     # XGBoost model implementation
├── src/                     # Source code
│   ├── api/                 # API implementation
│   ├── features/            # Feature engineering
│   └── visualization/       # Visualization components
├── test/                    # Test suite
```

## Development Workflow

### Branching Strategy

We follow a simplified Git flow model:

- `main`: Production-ready code
- `develop`: Integration branch for ongoing development
- `feature/*`: Feature branches for new functionality
- `bugfix/*`: Bug fix branches
- `docs/*`: Documentation updates

Always create your working branch from `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Commit Guidelines

Write clear, concise commit messages that explain the change and its purpose. Follow these guidelines:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Fix bug" not "Fixes bug")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

Example of a good commit message:
```
Add energy prediction feature for cold weather conditions

- Implement temperature coefficient model
- Add unit tests for temperature ranges
- Update documentation with new parameters

Fixes #123
```

### Pull Request Process

1. Update your branch with the latest changes from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout your-branch-name
   git rebase develop
   ```

2. Run tests and ensure code quality:
   ```bash
   # Run tests
   pytest
   
   # Run linters
   flake8 .
   black --check .
   ```

3. Push your branch to your fork:
   ```bash
   git push origin your-branch-name
   ```

4. Create a Pull Request against the `develop` branch of the main repository
5. Fill in the PR template with all required information
6. Request review from maintainers
7. Address any feedback from reviewers

## Coding Standards

### Python Style Guide

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some additional requirements:

- Use [Black](https://github.com/psf/black) for code formatting
- Use type hints for function parameters and return values
- Maximum line length is 88 characters (Black's default)
- Use docstrings for all public modules, functions, classes, and methods

Example function with proper formatting and docstrings:

```python
def predict_energy_consumption(
    route_data: Dict[str, Any], 
    vehicle_params: Dict[str, Any], 
    weather_conditions: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Predict energy consumption for an electric vehicle journey.
    
    Args:
        route_data: Dictionary containing route information
        vehicle_params: Dictionary containing vehicle specifications
        weather_conditions: Optional dictionary with weather data
        
    Returns:
        Dictionary containing energy consumption prediction results
    
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Function implementation
    ...
```

### Documentation

- Document all public modules, classes, methods, and functions
- Update the README.md when adding significant features
- Add explanatory comments for complex code sections
- Create or update examples when adding new functionality

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Structure tests by module/feature following the main project structure
- Use descriptive test names that explain what is being tested

Example test:

```python
def test_energy_prediction_with_cold_temperature():
    """Test that energy prediction increases appropriately in cold weather."""
    # Test setup
    route = {"distance": 100, "elevation_gain": 200}
    vehicle = {"efficiency": 0.16, "battery_capacity": 75.0}
    
    # Normal weather prediction
    normal_result = predict_energy_consumption(
        route, vehicle, {"temperature": 20.0}
    )
    
    # Cold weather prediction
    cold_result = predict_energy_consumption(
        route, vehicle, {"temperature": -10.0}
    )
    
    # Assert cold weather uses more energy
    assert cold_result["energy_kwh"] > normal_result["energy_kwh"]
    assert cold_result["range_km"] < normal_result["range_km"]
```

## Contribution Types

### Bug Reports

When reporting bugs:

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include detailed steps to reproduce
4. Provide environment information (OS, Python version, dependencies)
5. Include logs, error messages, and screenshots if applicable

### Feature Requests

When suggesting features:

1. Check existing issues and discussions
2. Use the feature request template
3. Clearly describe the problem your feature would solve
4. Outline the proposed functionality
5. Consider implementation complexity

### Code Contributions

Code contributions can include:

- New prediction models or improvements to existing ones
- Additional feature engineering approaches
- API enhancements
- Performance optimizations
- New visualizations or UI improvements

### Documentation Improvements

Documentation contributions are highly valued:

- Fixing typos and clarifying existing documentation
- Adding examples and tutorials
- Creating visual aids and diagrams
- Translating documentation to other languages

## Review Process

All contributions go through a review process:

1. Initial review by maintainers for approach and scope
2. Code review for quality, style, and correctness
3. Automated CI checks for tests, linting, and coverage
4. Final approval and merge by maintainers

Expect feedback and be prepared to make revisions to your contribution.

## Community

Join our community:

- Report issues and suggest features through GitHub Issues
- Discuss ideas and ask questions in GitHub Discussions
- Follow the project for updates

Thank you for contributing to the EV Energy Prediction project!