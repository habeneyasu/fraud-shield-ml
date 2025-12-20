# Contributing to Fraud Shield ML

Thank you for your interest in contributing to Fraud Shield ML! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/your-username/fraud-shield-ml.git
   cd fraud-shield-ml
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a new branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## 🔄 Development Workflow

### Typical Workflow

1. **Data Exploration** (`notebooks/eda-*.ipynb`)
   - Start with exploratory data analysis
   - Document findings and insights
   - Identify data quality issues

2. **Feature Engineering** (`notebooks/feature-engineering.ipynb`)
   - Create new features
   - Handle missing values and outliers
   - Encode categorical variables

3. **Modeling** (`notebooks/modeling.ipynb`)
   - Train and evaluate models
   - Compare different algorithms
   - Tune hyperparameters

4. **Explainability** (`notebooks/shap-explainability.ipynb`)
   - Analyze model decisions
   - Generate SHAP visualizations
   - Document feature importance

5. **Code Integration** (`src/`)
   - Convert notebook code to reusable modules
   - Add error handling and validation
   - Write unit tests

### Using Reusable Modules

The project includes reusable modules in `src/` that should be used for production code:

```python
from src.data_loader import load_fraud_data, save_dataframe
from src.preprocessing import handle_missing_values, scale_features, split_data
from src.modeling import train_model, evaluate_model, save_model

# Load data with error handling
df = load_fraud_data(data_dir='data', filename='Fraud_Data.csv')

# Preprocess data
df_clean = handle_missing_values(df, strategy='drop')
X_scaled, scaler = scale_features(X, scaler_type='standard')

# Train model
model = train_model(X_train, y_train, model_type='random_forest')

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
```

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all functions and classes (Google style)
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Error Handling

- Always include **try/except blocks** around I/O operations
- Add **validation checks** for critical processing steps
- Provide **clear error messages** with context
- Use **logging** instead of print statements for debugging

Example:
```python
def load_data(file_path: Path) -> pd.DataFrame:
    """Load data with error handling."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
```

### Code Organization

- **One function per task**: Keep functions focused
- **Reusable modules**: Place reusable code in `src/`
- **Notebooks for exploration**: Use notebooks for EDA and experimentation
- **Tests for modules**: Write tests for all functions in `src/`

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all functions in `src/` modules
- Place tests in the `tests/` directory
- Use descriptive test names: `test_function_name_scenario()`
- Test both success and failure cases

Example:
```python
def test_load_csv_success():
    """Test successful CSV loading."""
    df = load_csv('data/raw/test.csv')
    assert not df.empty
    assert len(df.columns) > 0

def test_load_csv_file_not_found():
    """Test CSV loading with missing file."""
    with pytest.raises(FileNotFoundError):
        load_csv('data/raw/nonexistent.csv')
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py -v
```

### Test Coverage

- Aim for **>80% code coverage** for `src/` modules
- Focus on testing critical paths and error handling
- Include edge cases and boundary conditions

## 🔀 Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features or bug fixes
3. **Run tests** and ensure they pass
4. **Check code style** (consider using `black` or `flake8`)
5. **Update CHANGELOG.md** if applicable

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Error handling included for I/O operations
- [ ] Validation checks added for critical steps
- [ ] Logging used instead of print statements
- [ ] Type hints added to functions
- [ ] Docstrings added to functions

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Testing
Describe how you tested your changes

## Related Issues
Closes #issue_number
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## 📁 Project Structure

### Directory Overview

```
fraud-shield-ml/
├── data/                    # Data files (gitignored)
│   ├── raw/                 # Original, unprocessed data
│   └── processed/           # Cleaned and feature-engineered data
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── eda-*.ipynb         # Exploratory data analysis
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
│
├── src/                     # Reusable Python modules
│   ├── data_loader.py      # Data loading functions
│   ├── preprocessing.py    # Preprocessing functions
│   ├── modeling.py         # Modeling functions
│   └── __init__.py
│
├── tests/                   # Unit tests
│   └── test_*.py
│
├── models/                  # Saved model artifacts
├── scripts/                 # Utility scripts
└── requirements.txt         # Python dependencies
```

### Where to Add Code

- **New reusable functions**: Add to appropriate module in `src/`
- **Exploratory analysis**: Create new notebook in `notebooks/`
- **Utility scripts**: Add to `scripts/`
- **Tests**: Add to `tests/` with `test_` prefix

### Module Responsibilities

- **`data_loader.py`**: All data loading and saving operations
- **`preprocessing.py`**: Data cleaning, transformation, and feature engineering
- **`modeling.py`**: Model training, evaluation, and prediction

## 🐛 Reporting Issues

When reporting bugs or suggesting features:

1. **Check existing issues** to avoid duplicates
2. **Use clear, descriptive titles**
3. **Provide context**: What were you trying to do?
4. **Include error messages** and stack traces
5. **Add steps to reproduce** if it's a bug
6. **Specify environment**: Python version, OS, etc.

## 📚 Additional Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Type Hints in Python](https://docs.python.org/3/library/typing.html)

## ❓ Questions?

If you have questions about contributing, please:
- Open an issue with the `question` label
- Check existing documentation
- Review example code in `src/example_usage.py`

---

Thank you for contributing to Fraud Shield ML! 🎉

