# Visualizations Directory

This directory contains visualization images referenced in the Interim Report 1.

## Required Visualizations

The following visualizations should be generated from the notebooks and saved here:

1. **class_distribution_comparison.png**
   - Source: `notebooks/eda-fraud-data.ipynb` and `notebooks/eda-creditcard.ipynb`
   - Description: Side-by-side bar charts comparing class distribution for e-commerce and banking datasets

2. **purchase_value_vs_fraud.png**
   - Source: `notebooks/eda-fraud-data.ipynb` or `src/analysis.py` (analyze_amount_vs_fraud function)
   - Description: Box plots and histograms comparing purchase value distributions for normal vs. fraudulent transactions

3. **global_fraud_heatmap.png**
   - Source: `notebooks/eda-fraud-data.ipynb` (Geolocation Integration section)
   - Description: Heatmap or bar chart showing fraud rates by country

4. **fraud_by_time_since_signup.png**
   - Source: `notebooks/eda-fraud-data.ipynb` (Feature Engineering section)
   - Description: Line chart or bar chart showing fraud rate by time-since-signup bins

5. **feature_correlation_heatmap.png**
   - Source: `notebooks/feature-engineering.ipynb` or `notebooks/eda-fraud-data.ipynb`
   - Description: Correlation heatmap matrix of all engineered features

## Generating Visualizations

To generate these visualizations, run the corresponding notebooks and save the figures:

```python
# Example: Save visualization in notebook
plt.savefig('visualizations/figure_name.png', dpi=300, bbox_inches='tight')
```

Or use the analysis module functions with save_path parameter:

```python
from src.analysis import analyze_amount_vs_fraud

results = analyze_amount_vs_fraud(
    df,
    amount_column='purchase_value',
    fraud_column='class',
    save_path='visualizations/purchase_value_vs_fraud.png'
)
```

## Image Specifications

- **Format**: PNG (recommended for reports)
- **Resolution**: 300 DPI minimum
- **Size**: Optimized for report viewing (typically 1200-1600px width)
- **Background**: White or transparent

