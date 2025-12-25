# Tests

This directory contains unit tests for the project modules.

## Structure

```
tests/
├── __init__.py
├── test_data_loader.py      # Tests for data loading functions
├── test_preprocessing.py    # Tests for preprocessing functions
├── test_modeling.py         # Tests for modeling functions
└── test_analysis.py         # Tests for analysis functions
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py -v
```

## Writing Tests

When adding new functionality:
1. Create corresponding test file: `test_<module_name>.py`
2. Follow pytest conventions
3. Test both success and error cases
4. Aim for >80% code coverage

## Best Practices

- Use descriptive test names: `test_function_name_scenario`
- Test edge cases and error handling
- Use fixtures for common setup
- Keep tests independent and isolated

