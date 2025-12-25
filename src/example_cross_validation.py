"""
Example: Using CrossValidator Class

This script demonstrates how to use the CrossValidator class to perform
Stratified K-Fold cross-validation on baseline and ensemble models for
fraud detection.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.baseline_model import BaselineModel
from src.ensemble_model import EnsembleModel
from src.cross_validation import CrossValidator, CrossValidationResults


def example_baseline_cross_validation():
    """Example: Cross-validation on baseline Logistic Regression model."""
    print("=" * 80)
    print("Example 1: Cross-Validation - Baseline Model (Logistic Regression)")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Prepare data
    print("\n2. Preparing data...")
    try:
        prep = DataPreparation(
            dataset_type='ecommerce',
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        split_result = prep.prepare_and_split(df)
        print(f"✓ Data prepared and split")
        print(f"  Training set: {split_result.train_size} samples")
        print(f"  Test set: {split_result.test_size} samples")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Create baseline model instance
    print("\n3. Creating baseline model instance...")
    baseline = BaselineModel(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    print("✓ Baseline model instance created")
    
    # Step 4: Perform cross-validation
    print("\n4. Performing 5-fold Stratified K-Fold cross-validation...")
    try:
        cv = CrossValidator(
            n_folds=5,
            random_state=42,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        results: CrossValidationResults = cv.cross_validate(
            baseline,
            split_result.X_train,
            split_result.y_train
        )
        
        print("\n✓ Cross-validation completed")
        
    except Exception as e:
        print(f"✗ Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    cv.print_summary(results)
    
    # Step 6: Access specific metrics
    print("\n📊 Key Metrics Summary:")
    print("-" * 80)
    print(f"F1-Score:     {results.metrics_summary['f1']['mean']:.4f} "
          f"(± {results.metrics_summary['f1']['std']:.4f})")
    print(f"AUC-PR:       {results.metrics_summary['pr_auc']['mean']:.4f} "
          f"(± {results.metrics_summary['pr_auc']['std']:.4f})")
    print(f"ROC-AUC:      {results.metrics_summary['roc_auc']['mean']:.4f} "
          f"(± {results.metrics_summary['roc_auc']['std']:.4f})")
    print(f"Precision:    {results.metrics_summary['precision']['mean']:.4f} "
          f"(± {results.metrics_summary['precision']['std']:.4f})")
    print(f"Recall:       {results.metrics_summary['recall']['mean']:.4f} "
          f"(± {results.metrics_summary['recall']['std']:.4f})")
    
    print("\n✓ Example 1 completed successfully!")


def example_ensemble_cross_validation():
    """Example: Cross-validation on ensemble Random Forest model."""
    print("\n" + "=" * 80)
    print("Example 2: Cross-Validation - Ensemble Model (Random Forest)")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Prepare data
    print("\n2. Preparing data...")
    try:
        prep = DataPreparation(
            dataset_type='ecommerce',
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        split_result = prep.prepare_and_split(df)
        print(f"✓ Data prepared and split")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Step 3: Create ensemble model instance (without hyperparameter tuning)
    print("\n3. Creating ensemble model instance...")
    ensemble = EnsembleModel(
        model_type='random_forest',
        class_weight='balanced',
        random_state=42
    )
    print("✓ Ensemble model instance created")
    
    # Step 4: Perform cross-validation
    print("\n4. Performing 5-fold Stratified K-Fold cross-validation...")
    try:
        cv = CrossValidator(
            n_folds=5,
            random_state=42,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        # Train model with default parameters first
        ensemble.train(
            split_result.X_train,
            split_result.y_train,
            n_estimators=100,
            max_depth=20
        )
        
        results: CrossValidationResults = cv.cross_validate(
            ensemble,
            split_result.X_train,
            split_result.y_train
        )
        
        print("\n✓ Cross-validation completed")
        
    except Exception as e:
        print(f"✗ Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    cv.print_summary(results)
    
    # Step 6: Access specific metrics
    print("\n📊 Key Metrics Summary:")
    print("-" * 80)
    print(f"F1-Score:     {results.metrics_summary['f1']['mean']:.4f} "
          f"(± {results.metrics_summary['f1']['std']:.4f})")
    print(f"AUC-PR:       {results.metrics_summary['pr_auc']['mean']:.4f} "
          f"(± {results.metrics_summary['pr_auc']['std']:.4f})")
    print(f"ROC-AUC:      {results.metrics_summary['roc_auc']['mean']:.4f} "
          f"(± {results.metrics_summary['roc_auc']['std']:.4f})")
    
    print("\n✓ Example 2 completed successfully!")


def example_banking_dataset_cross_validation():
    """Example: Cross-validation on banking dataset with XGBoost."""
    print("\n" + "=" * 80)
    print("Example 3: Cross-Validation - XGBoost on Banking Dataset")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = load_creditcard_data(data_dir='../data', filename='creditcard.csv')
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Prepare data
    print("\n2. Preparing data...")
    try:
        prep = DataPreparation(
            dataset_type='banking',
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        split_result = prep.prepare_and_split(df)
        print(f"✓ Data prepared and split")
        print(f"  Training set: {split_result.train_size} samples")
        print(f"  Test set: {split_result.test_size} samples")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Step 3: Create XGBoost model instance
    print("\n3. Creating XGBoost model instance...")
    ensemble = EnsembleModel(
        model_type='xgboost',
        class_weight='balanced',
        random_state=42
    )
    print("✓ XGBoost model instance created")
    
    # Step 4: Perform cross-validation
    print("\n4. Performing 5-fold Stratified K-Fold cross-validation...")
    try:
        cv = CrossValidator(
            n_folds=5,
            random_state=42,
            metrics=['f1', 'pr_auc', 'roc_auc', 'precision', 'recall']
        )
        
        # Train model with default parameters first
        ensemble.train(
            split_result.X_train,
            split_result.y_train,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
        results: CrossValidationResults = cv.cross_validate(
            ensemble,
            split_result.X_train,
            split_result.y_train
        )
        
        print("\n✓ Cross-validation completed")
        
    except Exception as e:
        print(f"✗ Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    cv.print_summary(results)
    
    print("\n✓ Example 3 completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Cross-Validation (Stratified K-Fold) - Usage Examples")
    print("=" * 80)
    print("\nThis script demonstrates:")
    print("  - Stratified K-Fold cross-validation (k=5)")
    print("  - Computing multiple metrics across folds")
    print("  - Reporting mean and standard deviation for each metric")
    print("  - Working with both baseline and ensemble models")
    print("  - Reliable performance estimation for model comparison")
    
    # Run examples
    example_baseline_cross_validation()
    example_ensemble_cross_validation()
    example_banking_dataset_cross_validation()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nKey Benefits of Cross-Validation:")
    print("  ✓ More reliable performance estimation than single train-test split")
    print("  ✓ Reduces variance in performance estimates")
    print("  ✓ Provides confidence intervals (mean ± std)")
    print("  ✓ Helps identify overfitting (high variance across folds)")
    print("  ✓ Better model selection and comparison")


if __name__ == '__main__':
    main()

