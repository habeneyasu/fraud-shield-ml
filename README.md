# Fraud Shield ML

A production-ready machine learning solution for fraud detection in e-commerce and banking transactions, featuring interpretable models, comprehensive evaluation, and explainable AI.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

Fraud Shield ML provides an end-to-end pipeline for building, training, and deploying fraud detection models. The project emphasizes **interpretability**, **reliability**, and **production-readiness** through object-oriented design, comprehensive error handling, and rigorous evaluation methodologies.

**Key Capabilities:**
- 🔍 **Data Analysis**: Risk pattern analysis with bivariate insights and statistical validation
- 🛠️ **Data Preprocessing**: Missing value handling, feature engineering, scaling, and resampling
- 🎯 **Stratified data preparation** with automatic feature-target separation
- 📊 **Baseline & ensemble models** (Logistic Regression, Random Forest, XGBoost, LightGBM)
- 🔧 **Hyperparameter tuning** with GridSearchCV and RandomizedSearchCV
- ✅ **Cross-validation** with Stratified K-Fold for reliable performance estimation
- 🏆 **Model comparison** considering both performance and interpretability
- 📈 **Comprehensive metrics** (AUC-PR, F1-Score, ROC-AUC, Confusion Matrix)

## Features

- **Object-Oriented Design**: Reusable, professional classes following industry best practices
- **Production-Ready**: Comprehensive error handling, validation, and logging
- **Data Analysis**: Risk pattern analysis with statistical validation and stakeholder-friendly interpretations
- **Advanced Preprocessing**: Feature engineering, scaling, encoding, and imbalanced data handling
- **Model Interpretability**: Built-in interpretability scoring and SHAP integration support
- **Comprehensive Evaluation**: Multiple metrics with cross-validation support
- **Modular Architecture**: Clean separation of concerns for maintainability

## Tech Stack

- **Python 3.10+**
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/habeneyasu/fraud-shield-ml.git
cd fraud-shield-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Analysis & Preprocessing

```python
from src.data_loader import load_fraud_data
from src.preprocessing import (
    handle_missing_values, remove_duplicates, encode_categorical,
    create_transaction_frequency_features, create_preprocessing_pipeline
)
from src.analysis import (
    analyze_amount_vs_fraud, analyze_device_vs_fraud,
    generate_risk_summary_report
)

# Load and clean data
df = load_fraud_data(data_dir='data', filename='Fraud_Data.csv')
df = handle_missing_values(df, strategy='drop')
df = remove_duplicates(df)

# Feature engineering
df = create_transaction_frequency_features(
    df, windows=['1h', '24h']  # Transaction velocity features
)

# Risk pattern analysis
amount_risk = analyze_amount_vs_fraud(df)
device_risk = analyze_device_vs_fraud(df, top_n=10)
risk_report = generate_risk_summary_report(df)

# Preprocessing pipeline
pipeline = create_preprocessing_pipeline(
    scaler_type='standard',
    resampling_strategy='smote',  # Handle class imbalance
    random_state=42
)
```

### Model Training

```python
from src.data_preparation import DataPreparation
from src.baseline_model import BaselineModel
from src.ensemble_model import EnsembleModel
from src.cross_validation import CrossValidator
from src.model_comparison import ModelComparator, ModelComparisonEntry

# Prepare data
prep = DataPreparation(dataset_type='ecommerce', test_size=0.2, random_state=42)
split_result = prep.prepare_and_split(df)

# Train baseline model
baseline = BaselineModel(class_weight='balanced', random_state=42)
baseline_results = baseline.train_and_evaluate(
    split_result.X_train, split_result.y_train,
    split_result.X_test, split_result.y_test
)

# Train ensemble model with hyperparameter tuning
ensemble = EnsembleModel(model_type='random_forest', class_weight='balanced')
ensemble_results = ensemble.train_and_evaluate(
    split_result.X_train, split_result.y_train,
    split_result.X_test, split_result.y_test,
    param_grid={'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    tune_hyperparameters=True, cv=5
)

# Cross-validation
cv = CrossValidator(n_folds=5, random_state=42)
cv_results = cv.cross_validate(baseline, split_result.X_train, split_result.y_train)

# Compare models
comparator = ModelComparator(primary_metric='f1', interpretability_weight=0.3)
comparison_results = comparator.compare_models([
    ModelComparisonEntry('Logistic Regression', 'baseline', 
                        baseline_results.test_metrics, interpretability_score=1.0),
    ModelComparisonEntry('Random Forest', 'random_forest',
                        ensemble_results.test_metrics, interpretability_score=0.6)
])
```

## Project Structure

```
fraud-shield-ml/
├── src/                          # Production-ready modules
│   ├── data_loader.py           # Data loading with validation
│   ├── preprocessing.py         # Preprocessing & feature engineering
│   ├── analysis.py              # Risk pattern analysis
│   ├── data_preparation.py      # OOP data splitting & feature separation
│   ├── baseline_model.py        # Logistic Regression baseline
│   ├── ensemble_model.py        # Random Forest, XGBoost, LightGBM
│   ├── cross_validation.py      # Stratified K-Fold CV
│   └── model_comparison.py      # Model comparison & selection
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # Unit tests (pytest)
├── models/                      # Saved model artifacts (.joblib, .pkl)
├── reports/                     # Generated comparison reports (text, JSON)
├── scripts/                     # Utility scripts
├── visualizations/              # Generated visualization images
└── requirements.txt            # Dependencies
```

### Directory Descriptions

- **`src/`**: Production-ready Python modules with comprehensive error handling. All reusable functionality lives here.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis, feature engineering, and experimentation.
- **`tests/`**: Unit tests for all modules. Run with `pytest tests/ -v`. See `tests/README.md` for details.
- **`models/`**: Saved trained model artifacts. Models are saved as `.joblib` or `.pkl` files. This directory is gitignored. See `models/README.md` for usage.
- **`reports/`**: Generated model comparison reports and analysis outputs. Contains text and JSON reports from comprehensive model comparison scripts. This directory is gitignored (except README.md). See `reports/README.md` for details.
- **`scripts/`**: Utility scripts for automation tasks (model comparison, visualization generation, etc.).
- **`visualizations/`**: Generated visualization images (plots, charts, heatmaps) from data analysis and model evaluation.
- **`data/`**: Data directory (gitignored). Raw data in `raw/`, processed data in `processed/`.

## Example Scripts

```bash
# Data preparation example
python src/example_data_preparation.py

# Baseline model training
python src/example_baseline_model.py

# Ensemble model with hyperparameter tuning
python src/example_ensemble_model.py

# Cross-validation
python src/example_cross_validation.py

# Model comparison
python src/example_model_comparison.py

# Comprehensive model comparison with detailed reporting
python scripts/comprehensive_model_comparison.py --dataset ecommerce
python scripts/comprehensive_model_comparison.py --dataset banking
```

## Data Analysis & Preprocessing

### Risk Pattern Analysis

The `analysis` module provides targeted bivariate analyses to identify high-risk patterns:

- **Amount vs Fraud**: Identifies high-risk transaction amounts
- **Device vs Fraud**: Detects devices with elevated fraud rates
- **Source vs Fraud**: Analyzes fraud rates by traffic source
- **Browser vs Fraud**: Identifies browsers linked to automated fraud

Each analysis provides:
- Statistical validation with significance testing
- Risk level classifications (CRITICAL, HIGH, MODERATE, LOW)
- Narrative interpretations for stakeholders
- Actionable recommendations

### Preprocessing Features

- **Missing Value Handling**: Multiple strategies (drop, fill, forward_fill)
- **Feature Engineering**: Transaction velocity features (1h, 24h windows)
- **Categorical Encoding**: Label encoding and one-hot encoding
- **Feature Scaling**: StandardScaler and MinMaxScaler
- **Imbalanced Data**: SMOTE, undersampling, and combined strategies
- **Reproducible Pipelines**: Scikit-learn compatible preprocessing pipelines

## Model Evaluation

The project emphasizes metrics critical for fraud detection:

- **AUC-PR** (Average Precision): Preferred for imbalanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall discriminative ability
- **Confusion Matrix**: Detailed classification breakdown

All models support cross-validation with mean ± standard deviation reporting for reliable performance estimation.

## Model Selection

The `ModelComparator` class enables intelligent model selection by:

- **Side-by-side comparison** of all models
- **Weighted scoring** balancing performance and interpretability
- **Automatic justification** for best model selection
- **Cross-validation integration** for robust evaluation

```python
# Performance-focused (70% performance, 30% interpretability)
comparator = ModelComparator(
    primary_metric='f1',
    interpretability_weight=0.3,
    performance_weight=0.7
)
```

### Comprehensive Model Comparison Script

The `scripts/comprehensive_model_comparison.py` script provides a complete workflow:

1. **Trains multiple models**: Baseline (Logistic Regression) and ensemble models (Random Forest, XGBoost, LightGBM)
2. **Performs cross-validation**: Stratified K-Fold (k=5) on all models with mean ± std reporting
3. **Side-by-side comparison**: Tabulates all metrics in a comprehensive table
4. **Programmatic documentation**: Generates detailed text and JSON reports
5. **Selection rationale**: Documents which model is selected and why, considering both performance and interpretability

**Generated Reports:**
- **Text Report**: Detailed human-readable report with justification and recommendations
- **JSON Summary**: Machine-readable summary for programmatic access

**Usage:**
```bash
# Run comprehensive comparison for e-commerce dataset
python scripts/comprehensive_model_comparison.py --dataset ecommerce

# Run for banking dataset
python scripts/comprehensive_model_comparison.py --dataset banking

# Specify output directory
python scripts/comprehensive_model_comparison.py --dataset ecommerce --output-dir reports
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of a training portfolio. See repository for license information.

## Author

**Haben Eyasu**

- GitHub: [@habeneyasu](https://github.com/habeneyasu)

---

**Note**: This project demonstrates production-ready machine learning practices for fraud detection, including object-oriented design, comprehensive evaluation, and model interpretability considerations.
