# Testing Instructions

## Running Tests

### Prerequisites
```bash
# Install pytest and coverage tools
pip install pytest pytest-cov

# Install development dependencies
pip install -e ".[dev]"
```

### Usage

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specifics
pytest test_core.py
pytest test_calibration.py
pytest test_pricing.py
pytest test_neural.py
pytest test_integration.py
pytest test_edge_cases.py
pytest -k "jax"   
pytest -k "torch"   

# Run tests with coverage
pytest --cov=spetro

# Run tests with cov report
pip install pytest-cov
pytest --cov=spetro --cov-report=html
```