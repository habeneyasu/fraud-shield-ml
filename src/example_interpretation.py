"""
Example: Feature Importance Interpretation

This script demonstrates how to interpret model predictions by:
- Comparing SHAP importance with built-in feature importance
- Identifying the top 5 drivers of fraud predictions
- Explaining surprising or counterintuitive findings

This addresses Task 3 - Model Explainability, Interpretation step.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.ensemble_model import EnsembleModel
from src.model_explainability import ModelExplainability


def example_ecommerce_interpretation():
    """
    Example: Interpretation for e-commerce fraud detection model.
    """
    print("=" * 80)
    print("FEATURE IMPORTANCE INTERPRETATION - E-COMMERCE DATASET")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading e-commerce fraud data...")
    try:
        df = load_fraud_data()
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Prepare data
    print("\n2. Preparing data (train-test split)...")
    try:
        prep = DataPreparation(dataset_type='ecommerce', random_state=42)
        split_result = prep.prepare_and_split(df)
        print(f"✓ Training set: {len(split_result.X_train)} samples")
        print(f"✓ Test set: {len(split_result.X_test)} samples")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Step 3: Train ensemble model
    print("\n3. Training Random Forest model...")
    try:
        ensemble = EnsembleModel(
            model_type='random_forest',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        ensemble.train(
            split_result.X_train,
            split_result.y_train,
            n_estimators=100,
            max_depth=20,
            min_samples_split=10
        )
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    # Step 4: Initialize explainer and extract built-in importance
    print("\n4. Extracting built-in feature importance...")
    try:
        explainer = ModelExplainability(
            model=ensemble.model,
            feature_names=split_result.X_train.columns.tolist(),
            random_state=42
        )
        
        builtin_importance = explainer.extract_feature_importance(method='auto')
        print(f"✓ Extracted built-in importance for {len(builtin_importance.importance_df)} features")
    except Exception as e:
        print(f"✗ Error extracting built-in importance: {e}")
        return
    
    # Step 5: Compute SHAP values
    print("\n5. Computing SHAP values (this may take a moment)...")
    try:
        shap_values, shap_explainer = explainer.compute_shap_values(
            split_result.X_test,
            sample_size=100,
            background_size=100
        )
        print(f"✓ SHAP values computed: shape {shap_values.shape}")
    except ImportError as e:
        print(f"✗ SHAP not installed: {e}")
        print("  Install with: pip install shap")
        return
    except Exception as e:
        print(f"✗ Error computing SHAP values: {e}")
        return
    
    # Step 6: Generate interpretation report
    print("\n6. Generating interpretation report...")
    try:
        report = explainer.generate_interpretation_report(
            builtin_importance,
            shap_values,
            split_result.X_test.iloc[:len(shap_values)],
            top_n=5,
            save_path='reports/interpretation_report_ecommerce.txt'
        )
        print("✓ Interpretation report generated")
        
        # Display summary
        print("\n" + report['summary'])
        
    except Exception as e:
        print(f"✗ Error generating interpretation report: {e}")
        return
    
    # Step 7: Display top 5 drivers
    print("\n7. Top 5 Drivers of Fraud Predictions:")
    print("-" * 80)
    print(report['top_drivers'].to_string())
    
    # Step 8: Display surprising findings
    print("\n8. Surprising Findings:")
    print("-" * 80)
    
    surprising = report['surprising_findings']
    
    if surprising['high_rank_difference']:
        print("\nHigh Rank Differences:")
        for finding in surprising['high_rank_difference'][:3]:
            print(f"  - {finding['feature']}: {finding['interpretation']}")
    
    if surprising['shap_higher']:
        print("\nFeatures More Important in SHAP:")
        for finding in surprising['shap_higher'][:3]:
            print(f"  - {finding['feature']}: {finding['interpretation']}")
    
    if surprising['builtin_higher']:
        print("\nFeatures More Important in Built-in:")
        for finding in surprising['builtin_higher'][:3]:
            print(f"  - {finding['feature']}: {finding['interpretation']}")
    
    # Step 9: Visualize comparison
    print("\n9. Creating importance comparison visualization...")
    try:
        comparison_path = explainer.visualize_importance_comparison(
            report['comparison_df'],
            top_n=10,
            save_path='visualizations/importance_comparison_ecommerce.png',
            title='Feature Importance Comparison - E-Commerce Fraud Detection\n(Built-in vs SHAP)'
        )
        print(f"✓ Comparison visualization saved to: {comparison_path}")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✓ Interpretation analysis completed successfully!")
    print("=" * 80)


def example_banking_interpretation():
    """
    Example: Interpretation for banking (credit card) fraud detection model.
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE INTERPRETATION - BANKING (CREDIT CARD) DATASET")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading credit card fraud data...")
    try:
        df = load_creditcard_data()
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Prepare data
    print("\n2. Preparing data (train-test split)...")
    try:
        prep = DataPreparation(dataset_type='banking', random_state=42)
        split_result = prep.prepare_and_split(df)
        print(f"✓ Training set: {len(split_result.X_train)} samples")
        print(f"✓ Test set: {len(split_result.X_test)} samples")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Step 3: Train ensemble model
    print("\n3. Training XGBoost model...")
    try:
        ensemble = EnsembleModel(
            model_type='xgboost',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        ensemble.train(
            split_result.X_train,
            split_result.y_train,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    # Step 4: Initialize explainer and extract built-in importance
    print("\n4. Extracting built-in feature importance...")
    try:
        explainer = ModelExplainability(
            model=ensemble.model,
            feature_names=split_result.X_train.columns.tolist(),
            random_state=42
        )
        
        builtin_importance = explainer.extract_feature_importance(method='auto')
        print(f"✓ Extracted built-in importance for {len(builtin_importance.importance_df)} features")
    except Exception as e:
        print(f"✗ Error extracting built-in importance: {e}")
        return
    
    # Step 5: Compute SHAP values
    print("\n5. Computing SHAP values (this may take a moment)...")
    try:
        shap_values, shap_explainer = explainer.compute_shap_values(
            split_result.X_test,
            sample_size=100,
            background_size=100
        )
        print(f"✓ SHAP values computed: shape {shap_values.shape}")
    except ImportError as e:
        print(f"✗ SHAP not installed: {e}")
        print("  Install with: pip install shap")
        return
    except Exception as e:
        print(f"✗ Error computing SHAP values: {e}")
        return
    
    # Step 6: Generate interpretation report
    print("\n6. Generating interpretation report...")
    try:
        report = explainer.generate_interpretation_report(
            builtin_importance,
            shap_values,
            split_result.X_test.iloc[:len(shap_values)],
            top_n=5,
            save_path='reports/interpretation_report_banking.txt'
        )
        print("✓ Interpretation report generated")
        
        # Display summary
        print("\n" + report['summary'])
        
    except Exception as e:
        print(f"✗ Error generating interpretation report: {e}")
        return
    
    # Step 7: Display top 5 drivers
    print("\n7. Top 5 Drivers of Fraud Predictions:")
    print("-" * 80)
    print(report['top_drivers'].to_string())
    
    # Step 8: Display surprising findings
    print("\n8. Surprising Findings:")
    print("-" * 80)
    
    surprising = report['surprising_findings']
    
    if surprising['high_rank_difference']:
        print("\nHigh Rank Differences:")
        for finding in surprising['high_rank_difference'][:3]:
            print(f"  - {finding['feature']}: {finding['interpretation']}")
    
    if surprising['shap_higher']:
        print("\nFeatures More Important in SHAP:")
        for finding in surprising['shap_higher'][:3]:
            print(f"  - {finding['feature']}: {finding['interpretation']}")
    
    if surprising['builtin_higher']:
        print("\nFeatures More Important in Built-in:")
        for finding in surprising['builtin_higher'][:3]:
            print(f"  - {finding['feature']}: {finding['interpretation']}")
    
    # Step 9: Visualize comparison
    print("\n9. Creating importance comparison visualization...")
    try:
        comparison_path = explainer.visualize_importance_comparison(
            report['comparison_df'],
            top_n=10,
            save_path='visualizations/importance_comparison_banking.png',
            title='Feature Importance Comparison - Credit Card Fraud Detection\n(Built-in vs SHAP)'
        )
        print(f"✓ Comparison visualization saved to: {comparison_path}")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✓ Interpretation analysis completed successfully!")
    print("=" * 80)


def main():
    """Run all interpretation examples."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE INTERPRETATION")
    print("Task 3 - Model Explainability, Interpretation")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  - Comparing SHAP importance with built-in feature importance")
    print("  - Identifying the top 5 drivers of fraud predictions")
    print("  - Explaining surprising or counterintuitive findings")
    print("\nNote: SHAP must be installed (pip install shap)")
    print("=" * 80)
    
    # Run e-commerce example
    example_ecommerce_interpretation()
    
    # Run banking example
    example_banking_interpretation()
    
    print("\n" + "=" * 80)
    print("ALL INTERPRETATION EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - reports/interpretation_report_ecommerce.txt")
    print("  - reports/interpretation_report_banking.txt")
    print("  - visualizations/importance_comparison_ecommerce.png")
    print("  - visualizations/importance_comparison_banking.png")
    print("\nNext steps:")
    print("  - Review interpretation reports for business insights")
    print("  - Use findings to improve fraud detection strategies")


if __name__ == '__main__':
    main()

