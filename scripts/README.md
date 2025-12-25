# Scripts

This directory contains utility scripts for data processing, model training, and other automation tasks.

## Available Scripts

### `comprehensive_model_comparison.py`

Comprehensive model comparison script that trains multiple models, performs cross-validation, compares them side-by-side, and generates detailed reports documenting model selection.

**Features:**
- Trains baseline (Logistic Regression) and ensemble models (Random Forest, XGBoost, LightGBM)
- Performs Stratified K-Fold cross-validation (k=5) on all models
- Tabulates metrics side-by-side for easy comparison
- Generates detailed text and JSON reports
- Documents model selection rationale considering performance and interpretability

**Usage:**
```bash
# E-commerce dataset
python scripts/comprehensive_model_comparison.py --dataset ecommerce

# Banking dataset
python scripts/comprehensive_model_comparison.py --dataset banking

# Custom output directory
python scripts/comprehensive_model_comparison.py --dataset ecommerce --output-dir reports

# Quiet mode (suppress progress output)
python scripts/comprehensive_model_comparison.py --dataset ecommerce --quiet
```

**Output:**
- Text report: `reports/model_comparison_{dataset}_{timestamp}.txt`
- JSON summary: `reports/model_comparison_{dataset}_{timestamp}.json`

### `generate_visualizations.py`

Generates visualizations for data analysis reports.

**Usage:**
```bash
python scripts/generate_visualizations.py
```

## Usage

Scripts in this directory can be run from the project root:

```bash
python scripts/script_name.py
```

