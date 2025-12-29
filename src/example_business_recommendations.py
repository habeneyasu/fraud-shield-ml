"""
Example: Business Recommendations Based on SHAP Analysis

This script demonstrates how to generate actionable business recommendations
based on SHAP insights from fraud detection models.

This addresses Task 3 - Model Explainability, Business Recommendations step.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.ensemble_model import EnsembleModel
from src.model_explainability import ModelExplainability


def example_ecommerce_recommendations():
    """
    Example: Business recommendations for e-commerce fraud detection.
    """
    print("=" * 80)
    print("BUSINESS RECOMMENDATIONS - E-COMMERCE DATASET")
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
        print(f"✓ Extracted built-in importance")
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
        interpretation_report = explainer.generate_interpretation_report(
            builtin_importance,
            shap_values,
            split_result.X_test.iloc[:len(shap_values)],
            top_n=5
        )
        print("✓ Interpretation report generated")
    except Exception as e:
        print(f"✗ Error generating interpretation report: {e}")
        return
    
    # Step 7: Generate business recommendations
    print("\n7. Generating business recommendations from SHAP insights...")
    try:
        recommendations = explainer.generate_business_recommendations(
            interpretation_report,
            dataset_type='ecommerce',
            min_recommendations=3
        )
        print(f"✓ Generated {len(recommendations)} business recommendations")
    except Exception as e:
        print(f"✗ Error generating recommendations: {e}")
        return
    
    # Step 8: Display recommendations
    print("\n8. Business Recommendations:")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']} PRIORITY]")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"\n   SHAP Insight: {rec['shap_insight']}")
        print(f"   Related Feature: {rec['feature']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print("\n" + "-" * 80)
    
    # Step 9: Format and save recommendations report
    print("\n9. Saving recommendations report...")
    try:
        report_text = explainer.format_recommendations_report(
            recommendations,
            save_path='reports/business_recommendations_ecommerce.txt'
        )
        print("✓ Recommendations report saved")
        print("\n" + report_text)
    except Exception as e:
        print(f"✗ Error saving report: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✓ Business recommendations analysis completed successfully!")
    print("=" * 80)


def example_banking_recommendations():
    """
    Example: Business recommendations for banking (credit card) fraud detection.
    """
    print("\n" + "=" * 80)
    print("BUSINESS RECOMMENDATIONS - BANKING (CREDIT CARD) DATASET")
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
        print(f"✓ Extracted built-in importance")
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
        interpretation_report = explainer.generate_interpretation_report(
            builtin_importance,
            shap_values,
            split_result.X_test.iloc[:len(shap_values)],
            top_n=5
        )
        print("✓ Interpretation report generated")
    except Exception as e:
        print(f"✗ Error generating interpretation report: {e}")
        return
    
    # Step 7: Generate business recommendations
    print("\n7. Generating business recommendations from SHAP insights...")
    try:
        recommendations = explainer.generate_business_recommendations(
            interpretation_report,
            dataset_type='banking',
            min_recommendations=3
        )
        print(f"✓ Generated {len(recommendations)} business recommendations")
    except Exception as e:
        print(f"✗ Error generating recommendations: {e}")
        return
    
    # Step 8: Display recommendations
    print("\n8. Business Recommendations:")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']} PRIORITY]")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"\n   SHAP Insight: {rec['shap_insight']}")
        print(f"   Related Feature: {rec['feature']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print("\n" + "-" * 80)
    
    # Step 9: Format and save recommendations report
    print("\n9. Saving recommendations report...")
    try:
        report_text = explainer.format_recommendations_report(
            recommendations,
            save_path='reports/business_recommendations_banking.txt'
        )
        print("✓ Recommendations report saved")
        print("\n" + report_text)
    except Exception as e:
        print(f"✗ Error saving report: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✓ Business recommendations analysis completed successfully!")
    print("=" * 80)


def main():
    """Run all business recommendations examples."""
    print("\n" + "=" * 80)
    print("BUSINESS RECOMMENDATIONS BASED ON SHAP ANALYSIS")
    print("Task 3 - Model Explainability, Business Recommendations")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  - Generating actionable business recommendations from SHAP insights")
    print("  - Connecting recommendations to specific SHAP findings")
    print("  - Prioritizing recommendations based on expected impact")
    print("\nNote: SHAP must be installed (pip install shap)")
    print("=" * 80)
    
    # Run e-commerce example
    example_ecommerce_recommendations()
    
    # Run banking example
    example_banking_recommendations()
    
    print("\n" + "=" * 80)
    print("ALL BUSINESS RECOMMENDATIONS EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - reports/business_recommendations_ecommerce.txt")
    print("  - reports/business_recommendations_banking.txt")
    print("\nNext steps:")
    print("  - Review recommendations with business stakeholders")
    print("  - Prioritize implementation based on expected impact")
    print("  - Monitor effectiveness of implemented recommendations")


if __name__ == '__main__':
    main()

