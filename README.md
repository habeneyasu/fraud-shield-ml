# Fraud Shield ML

**A comprehensive machine learning solution for fraud detection in e-commerce and banking transactions.**

Fraud Shield ML is an advanced machine learning project designed to detect and prevent fraudulent activities across e-commerce platforms and banking systems. This project leverages state-of-the-art ML algorithms, feature engineering techniques, and explainable AI to build robust fraud detection models that can identify suspicious transactions in real-time.

## 🎯 Overview

Fraud Shield ML provides a complete end-to-end pipeline for fraud detection, from exploratory data analysis to model deployment. The project is specifically tailored for:

- **E-commerce Fraud Detection**: Identifying fraudulent online transactions, payment fraud, and account takeovers
- **Banking Fraud Detection**: Detecting credit card fraud, unauthorized transactions, and suspicious banking activities

## ✨ Features

- **Comprehensive EDA**: Detailed exploratory data analysis for fraud datasets
- **Advanced Feature Engineering**: Automated feature creation and selection
- **Multiple ML Models**: Support for XGBoost, LightGBM, and scikit-learn algorithms
- **Model Explainability**: SHAP-based interpretability for transparent fraud detection
- **Production-Ready**: Modular code structure for easy deployment
- **CI/CD Integration**: Automated testing with GitHub Actions
- **Documentation**: Well-documented notebooks and codebase

## 🛠️ Technologies

- **Python 3.12+**
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest

## 📁 Project Structure

```
├── .vscode/
│   └── settings.json              # VS Code configuration
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Original datasets
│   └── processed/                 # Cleaned and feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb       # EDA for fraud datasets
│   ├── eda-creditcard.ipynb       # EDA for credit card data
│   ├── feature-engineering.ipynb  # Feature engineering pipeline
│   ├── modeling.ipynb             # Model training and evaluation
│   ├── shap-explainability.ipynb  # SHAP-based model explainability
│   └── README.md
├── src/                           # Source code modules
│   └── __init__.py
├── tests/                         # Unit tests
│   └── __init__.py
├── models/                        # Saved model artifacts
├── scripts/                       # Utility scripts
│   ├── __init__.py
│   └── README.md
├── requirements.txt               # Python dependencies
├── README.md
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/habeneyasu/fraud-shield-ml.git
   cd fraud-shield-ml
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Jupyter (for notebooks)**
   ```bash
   jupyter notebook
   ```

## 📊 Usage

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

## 🔍 Model Explainability

This project emphasizes model interpretability using SHAP (SHapley Additive exPlanations) values. The `shap-explainability.ipynb` notebook demonstrates how to:

- Understand feature importance
- Explain individual predictions
- Visualize model decision-making process
- Build trust in fraud detection systems

## 📈 Performance Metrics

The project focuses on metrics critical for fraud detection:

- **Precision & Recall**: Balancing false positives and false negatives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall model performance
- **PR-AUC**: Performance on imbalanced datasets
- **Confusion Matrix**: Detailed classification breakdown

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is part of a training portfolio. Please refer to the repository for license information.

## 👤 Author

**Haben Eyasu**

- GitHub: [@habeneyasu](https://github.com/habeneyasu)

## 🙏 Acknowledgments

- Built as part of the KAIM Training Portfolio
- Designed for real-world e-commerce and banking fraud detection scenarios

---

**Note**: This project is designed for educational and portfolio purposes, demonstrating expertise in machine learning, fraud detection, and explainable AI for e-commerce and banking applications.
