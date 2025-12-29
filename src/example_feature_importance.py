"""
Example: Feature Importance Extraction and Visualization

This script demonstrates how to extract and visualize feature importance
from trained ensemble models using the ModelExplainability class.

This addresses Task 3 - Model Explainability, Instruction 1:
- Extract built-in feature importance from ensemble model
- Visualize the top 10 most important features
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.ensemble_model import EnsembleModel
from src.model_explainability import ModelExplainability


def example_ecommerce_feature_importance():
    """
    Example: Extract and visualize feature importance for e-commerce dataset.
    """
    print("=" * 80)
    print("FEATURE IMPORTANCE EXAMPLE - E-COMMERCE DATASET")
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
        print(f"✓ Features: {len(split_result.X_train.columns)}")
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
        
        # Train with basic hyperparameters (quick training for example)
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
    
    # Step 4: Extract feature importance
    print("\n4. Extracting feature importance...")
    try:
        explainer = ModelExplainability(
            model=ensemble.model,
            feature_names=split_result.X_train.columns.tolist(),
            random_state=42
        )
        
        importance_results = explainer.extract_feature_importance(method='auto')
        print(f"✓ Extracted importance for {len(importance_results.importance_df)} features")
        print(f"✓ Method used: {importance_results.method}")
    except Exception as e:
        print(f"✗ Error extracting feature importance: {e}")
        return
    
    # Step 5: Display top 10 features
    print("\n5. Top 10 Most Important Features:")
    print("-" * 80)
    top_features = explainer.get_top_features(importance_results, top_n=10)
    print(top_features.to_string())
    
    # Step 6: Print summary
    print("\n6. Feature Importance Summary:")
    explainer.print_feature_importance_summary(importance_results, top_n=10)
    
    # Step 7: Visualize top 10 features
    print("\n7. Creating visualization...")
    try:
        viz_path = explainer.visualize_feature_importance(
            importance_results,
            top_n=10,
            figsize=(12, 7),
            save_path='visualizations/feature_importance_ecommerce_top10.png',
            title='Top 10 Most Important Features - E-Commerce Fraud Detection\n(Random Forest)'
        )
        print(f"✓ Visualization saved to: {viz_path}")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✓ Feature importance example completed successfully!")
    print("=" * 80)


def example_banking_feature_importance():
    """
    Example: Extract and visualize feature importance for banking (credit card) dataset.
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE EXAMPLE - BANKING (CREDIT CARD) DATASET")
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
        print(f"✓ Features: {len(split_result.X_train.columns)}")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Step 3: Train ensemble model (XGBoost)
    print("\n3. Training XGBoost model...")
    try:
        ensemble = EnsembleModel(
            model_type='xgboost',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train with basic hyperparameters
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
    
    # Step 4: Extract feature importance
    print("\n4. Extracting feature importance...")
    try:
        explainer = ModelExplainability(
            model=ensemble.model,
            feature_names=split_result.X_train.columns.tolist(),
            random_state=42
        )
        
        importance_results = explainer.extract_feature_importance(method='auto')
        print(f"✓ Extracted importance for {len(importance_results.importance_df)} features")
        print(f"✓ Method used: {importance_results.method}")
    except Exception as e:
        print(f"✗ Error extracting feature importance: {e}")
        return
    
    # Step 5: Display top 10 features
    print("\n5. Top 10 Most Important Features:")
    print("-" * 80)
    top_features = explainer.get_top_features(importance_results, top_n=10)
    print(top_features.to_string())
    
    # Step 6: Print summary
    print("\n6. Feature Importance Summary:")
    explainer.print_feature_importance_summary(importance_results, top_n=10)
    
    # Step 7: Visualize top 10 features
    print("\n7. Creating visualization...")
    try:
        viz_path = explainer.visualize_feature_importance(
            importance_results,
            top_n=10,
            figsize=(12, 7),
            save_path='visualizations/feature_importance_banking_top10.png',
            title='Top 10 Most Important Features - Credit Card Fraud Detection\n(XGBoost)'
        )
        print(f"✓ Visualization saved to: {viz_path}")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✓ Feature importance example completed successfully!")
    print("=" * 80)


def main():
    """Run all feature importance examples."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE EXTRACTION AND VISUALIZATION")
    print("Task 3 - Model Explainability, Instruction 1")
    print("=" * 80)
    
    # Run e-commerce example
    example_ecommerce_feature_importance()
    
    # Run banking example
    example_banking_feature_importance()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Review visualizations in visualizations/ directory")
    print("  - Analyze top features for business insights")
    print("  - Proceed to SHAP analysis (Instruction 2)")


if __name__ == '__main__':
    main()

