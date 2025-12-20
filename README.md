# Fraud Shield ML

**A comprehensive machine learning solution for fraud detection in e-commerce and banking transactions.**

Fraud Shield ML is an advanced machine learning project designed to detect and prevent fraudulent activities across e-commerce platforms and banking systems. This project leverages state-of-the-art ML algorithms, feature engineering techniques, and explainable AI to build robust fraud detection models that can identify suspicious transactions in real-time.

## 🎯 Overview

Fraud Shield ML provides a complete end-to-end pipeline for fraud detection, from exploratory data analysis to model deployment. The project is specifically tailored for:

- **E-commerce Fraud Detection**: Identifying fraudulent online transactions, payment fraud, and account takeovers
- **Banking Fraud Detection**: Detecting credit card fraud, unauthorized transactions, and suspicious banking activities

## ✨ Features

- **Comprehensive EDA**: Detailed exploratory data analysis for fraud datasets
- **Targeted Risk Analysis**: Bivariate analyses focusing on high-risk patterns (amount, device, source, browser) with narrative interpretations for stakeholders
- **Advanced Feature Engineering**: Automated feature creation and selection
- **Multiple ML Models**: Support for XGBoost, LightGBM, Random Forest, and scikit-learn algorithms
- **Model Explainability**: SHAP-based interpretability for transparent fraud detection
- **Production-Ready**: Modular code structure with comprehensive error handling and validation
- **CI/CD Integration**: Automated testing with GitHub Actions
- **Comprehensive Documentation**: Well-documented notebooks, modules, and contribution guidelines

## 🛠️ Technologies

- **Python 3.12+**
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest

## 📁 Project Structure

```
fraud-shield-ml/
├── .vscode/                       # VS Code configuration
│   └── settings.json
│
├── .github/                       # GitHub configuration
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline for automated testing
│
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Original, unprocessed datasets
│   │   ├── Fraud_Data.csv         # E-commerce fraud dataset
│   │   ├── creditcard.csv         # Credit card fraud dataset
│   │   └── IpAddress_to_Country.csv
│   └── processed/                 # Cleaned and feature-engineered data
│       ├── fraud_data_cleaned.csv
│       └── creditcard_cleaned.csv
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── eda-fraud-data.ipynb       # EDA for e-commerce fraud data
│   ├── eda-creditcard.ipynb       # EDA for credit card data
│   ├── feature-engineering.ipynb  # Feature engineering pipeline
│   ├── modeling.ipynb             # Model training and evaluation
│   ├── shap-explainability.ipynb  # SHAP-based model explainability
│   └── README.md                  # Notebook documentation
│
├── src/                           # Reusable Python modules (production code)
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Data loading with error handling
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── modeling.py                # Model training and evaluation
│   ├── analysis.py                # Bivariate risk analysis with narrative interpretations
│   ├── example_usage.py           # Example workflow demonstration
│   └── example_bivariate_analysis.py  # Bivariate analysis examples
│
├── tests/                         # Unit tests
│   └── __init__.py
│
├── models/                        # Saved model artifacts (gitignored)
│
├── scripts/                       # Utility scripts
│   ├── __init__.py
│   └── README.md
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── CONTRIBUTING.md                # Contribution guidelines
└── .gitignore                     # Git ignore rules
```

### Directory Descriptions

- **`data/`**: Contains all datasets. Raw data goes in `raw/`, processed data in `processed/`. This directory is gitignored to avoid committing large files.
- **`notebooks/`**: Jupyter notebooks for exploratory analysis, experimentation, and visualization. Use these for EDA and initial model development.
- **`src/`**: Production-ready, reusable Python modules with comprehensive error handling and validation. Includes data loading, preprocessing, modeling, and risk analysis capabilities. These modules should be used for any production code or scripts.
- **`tests/`**: Unit tests for modules in `src/`. Run with `pytest`.
- **`models/`**: Saved model artifacts (`.joblib` or `.pkl` files). Gitignored to avoid committing large model files.
- **`scripts/`**: Utility scripts for data processing, model deployment, or automation tasks.

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+** (recommended) or Python 3.10+
- **pip** package manager
- **Git** for version control (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/habeneyasu/fraud-shield-ml.git
   cd fraud-shield-ml
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "from src import load_fraud_data, analyze_amount_vs_fraud; print('✓ Installation successful')"
   ```

5. **Set up Jupyter (for notebooks)**
   ```bash
   pip install jupyter
   jupyter notebook
   ```

## 📊 Usage

### Typical Workflow

1. **Exploratory Data Analysis** (`notebooks/eda-*.ipynb`)
   - Load and explore raw datasets
   - Identify data quality issues
   - Understand feature distributions
   - Analyze class imbalance

2. **Risk Pattern Analysis** (`src/analysis.py` or `src/example_bivariate_analysis.py`)
   - Perform bivariate analyses on high-risk patterns
   - Identify key risk drivers (amount, device, source, browser)
   - Generate stakeholder-friendly reports with narrative interpretations
   - Prioritize fraud prevention actions

3. **Feature Engineering** (`notebooks/feature-engineering.ipynb`)
   - Create new features based on risk analysis insights
   - Handle missing values and outliers
   - Encode categorical variables
   - Scale numerical features

4. **Model Development** (`notebooks/modeling.ipynb`)
   - Train multiple models (Random Forest, XGBoost, LightGBM)
   - Evaluate performance metrics
   - Tune hyperparameters
   - Compare model performance

5. **Model Explainability** (`notebooks/shap-explainability.ipynb`)
   - Analyze feature importance
   - Explain individual predictions
   - Generate SHAP visualizations

6. **Production Deployment** (`src/`)
   - Use production-ready modules with error handling
   - Write unit tests
   - Deploy models with monitoring

### Using Reusable Modules

The project includes production-ready modules in `src/` with comprehensive error handling and validation:

#### Data Loading and Preprocessing

```python
from src.data_loader import load_fraud_data, save_dataframe
from src.preprocessing import handle_missing_values, scale_features, split_data
from src.modeling import train_model, evaluate_model, save_model

# Load data with automatic error handling
df = load_fraud_data(data_dir='data', filename='Fraud_Data.csv')

# Preprocess with validation
df_clean = handle_missing_values(df, strategy='drop')
X_scaled, scaler = scale_features(X, scaler_type='standard')

# Split data
X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.2)

# Train model
model = train_model(X_train, y_train, model_type='random_forest')

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"F1 Score: {metrics['f1']:.4f}")

# Save model
save_model(model, 'models/fraud_model.joblib')
```

#### Risk Analysis and Bivariate Patterns

The analysis module provides targeted bivariate analyses with narrative interpretations for stakeholders:

```python
from src.analysis import (
    analyze_amount_vs_fraud,
    analyze_device_vs_fraud,
    analyze_source_vs_fraud,
    analyze_browser_vs_fraud,
    generate_risk_summary_report
)

# Analyze transaction amount risk
amount_results = analyze_amount_vs_fraud(
    df, 
    amount_column='purchase_value',
    fraud_column='class'
)
# Returns: risk level, interpretation, statistics, and visualizations

# Analyze device-level risk patterns
device_results = analyze_device_vs_fraud(
    df,
    device_column='device_id',
    fraud_column='class',
    top_n=10
)

# Generate comprehensive risk summary report
risk_report = generate_risk_summary_report(
    df,
    amount_column='purchase_value',
    device_column='device_id',
    source_column='source',
    browser_column='browser',
    fraud_column='class'
)
```

Each analysis function provides:
- **Visualizations**: Box plots, histograms, bar charts, and scatter plots
- **Statistical Validation**: Statistical tests and significance metrics
- **Risk Level Assessment**: CRITICAL, HIGH, MODERATE, or LOW risk classifications
- **Narrative Interpretations**: Stakeholder-friendly explanations of findings
- **Actionable Recommendations**: Specific, prioritized actions for fraud prevention

See `src/example_usage.py` for a complete modeling workflow and `src/example_bivariate_analysis.py` for risk analysis examples.

### Running Analysis Scripts

Execute the example scripts to see the modules in action:

```bash
# Run complete modeling workflow example
python src/example_usage.py

# Run bivariate risk analysis example
python src/example_bivariate_analysis.py
```

### Running Notebooks

Navigate to the `notebooks/` directory and open the Jupyter notebooks:

- **EDA Notebooks**: Start with `eda-fraud-data.ipynb` or `eda-creditcard.ipynb` to explore your datasets
- **Feature Engineering**: Use `feature-engineering.ipynb` to create and select features
- **Modeling**: Train models using `modeling.ipynb`
- **Explainability**: Analyze model decisions with `shap-explainability.ipynb`

### Running Tests

Execute the test suite:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🔍 Model Explainability & Risk Analysis

### Model Explainability

This project emphasizes model interpretability using SHAP (SHapley Additive exPlanations) values. The `shap-explainability.ipynb` notebook demonstrates how to:

- Understand feature importance
- Explain individual predictions
- Visualize model decision-making process
- Build trust in fraud detection systems

### Risk Analysis

The project includes comprehensive bivariate risk analysis capabilities to identify high-risk patterns:

- **Amount vs Fraud**: Identifies high-risk transaction amounts requiring additional scrutiny
- **Device vs Fraud**: Detects devices associated with elevated fraud rates
- **Source vs Fraud**: Analyzes fraud rates by traffic source (SEO, Ads, Direct, etc.)
- **Browser vs Fraud**: Identifies browsers linked to potential bot activity or automated fraud

Each analysis provides:
- Statistical validation with significance testing
- Risk level classifications (CRITICAL, HIGH, MODERATE, LOW)
- Narrative interpretations for stakeholders
- Actionable recommendations for fraud prevention
- Professional visualizations for presentations

## 📈 Performance Metrics

The project focuses on metrics critical for fraud detection:

- **Precision & Recall**: Balancing false positives and false negatives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall model performance
- **PR-AUC**: Performance on imbalanced datasets
- **Confusion Matrix**: Detailed classification breakdown

### Risk Analysis Metrics

The risk analysis module provides additional metrics for stakeholder reporting:

- **Risk Ratios**: Comparative analysis of fraud rates across different dimensions
- **Statistical Significance**: P-values and confidence intervals for risk patterns
- **Risk Level Classifications**: Standardized risk assessments for prioritization
- **Volume Impact**: Transaction volume analysis for high-risk patterns

## 🤝 Contributing

Contributions are welcome! We appreciate your help in making Fraud Shield ML better.

Please read our [Contributing Guidelines](CONTRIBUTING.md) for comprehensive details on:
- **Development Workflow**: How to set up your development environment
- **Coding Standards**: Code style, documentation, and best practices
- **Testing Requirements**: Writing and running tests
- **Pull Request Process**: How to submit contributions
- **Project Structure**: Where to add new code and features

**Quick Start for Contributors:**

1. **Fork the repository** and clone your fork
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with proper error handling, validation, and tests
4. **Follow coding standards** (see CONTRIBUTING.md)
5. **Write/update tests** to ensure your code works correctly
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request** with a clear description of changes

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📝 License

This project is part of a training portfolio. Please refer to the repository for license information.

## 👤 Author

**Haben Eyasu**

- GitHub: [@habeneyasu](https://github.com/habeneyasu)


## 📚 Additional Resources

- **Documentation**: See individual module docstrings for detailed function documentation
- **Examples**: Check `src/example_usage.py` and `src/example_bivariate_analysis.py` for usage examples
- **Contributing**: Read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- **Notebooks**: Explore Jupyter notebooks in `notebooks/` for detailed analysis workflows

---

**Note**: This project is designed for educational and portfolio purposes, demonstrating expertise in machine learning, fraud detection, explainable AI, and risk analysis for e-commerce and banking applications. All code includes comprehensive error handling, validation checks, and is production-ready.
