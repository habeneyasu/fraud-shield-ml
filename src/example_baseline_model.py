"""
Example: Using BaselineModel Class

This script demonstrates how to use the BaselineModel class to train and evaluate
a Logistic Regression baseline model for fraud detection on both e-commerce and
banking datasets.
"""

from pathlib import Path
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.baseline_model import BaselineModel, BaselineModelResults


def example_ecommerce_baseline():
    """Example using e-commerce fraud dataset."""
    print("=" * 80)
    print("Example 1: Baseline Model - E-commerce Dataset (Fraud_Data.csv)")
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
    
    # Step 3: Train and evaluate baseline model
    print("\n3. Training and evaluating baseline model...")
    try:
        baseline = BaselineModel(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        results: BaselineModelResults = baseline.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        print("\n✓ Baseline model training completed")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Display results
    print("\n" + "=" * 80)
    print("BASELINE MODEL RESULTS - E-COMMERCE DATASET")
    print("=" * 80)
    
    print("\n📊 Training Set Metrics:")
    for metric, value in results.train_metrics.items():
        if value is not None:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n📊 Test Set Metrics:")
    for metric, value in results.test_metrics.items():
        if value is not None:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n📊 Key Metrics (Test Set):")
    print(f"  F1-Score: {results.test_metrics['f1']:.4f}")
    print(f"  AUC-PR: {results.test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC: {results.test_metrics['roc_auc']:.4f}")
    
    print("\n📊 Confusion Matrix (Test Set):")
    print("  Format: [[TN, FP], [FN, TP]]")
    print(f"  {results.confusion_matrix}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = results.confusion_matrix.ravel()
    print(f"\n  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    
    print("\n📊 Classification Report (Test Set):")
    print(results.classification_report)
    
    # Step 5: Feature importance
    print("\n5. Feature importance (Top 10):")
    try:
        importance_df = baseline.get_feature_importance()
        if importance_df is not None:
            print(importance_df.head(10).to_string(index=False))
        else:
            print("  Feature importance not available")
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")
    
    # Step 6: Save model
    print("\n6. Saving model...")
    try:
        model_path = Path('../models') / 'baseline_ecommerce_model.joblib'
        baseline.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    print("\n✓ Example 1 completed successfully!")


def example_banking_baseline():
    """Example using banking credit card fraud dataset."""
    print("\n" + "=" * 80)
    print("Example 2: Baseline Model - Banking Dataset (creditcard.csv)")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = load_creditcard_data(data_dir='../data', filename='creditcard.csv')
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
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
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Train and evaluate baseline model
    print("\n3. Training and evaluating baseline model...")
    try:
        baseline = BaselineModel(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        results: BaselineModelResults = baseline.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        print("\n✓ Baseline model training completed")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Display results
    print("\n" + "=" * 80)
    print("BASELINE MODEL RESULTS - BANKING DATASET")
    print("=" * 80)
    
    print("\n📊 Training Set Metrics:")
    for metric, value in results.train_metrics.items():
        if value is not None:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n📊 Test Set Metrics:")
    for metric, value in results.test_metrics.items():
        if value is not None:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n📊 Key Metrics (Test Set):")
    print(f"  F1-Score: {results.test_metrics['f1']:.4f}")
    print(f"  AUC-PR: {results.test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC: {results.test_metrics['roc_auc']:.4f}")
    
    print("\n📊 Confusion Matrix (Test Set):")
    print("  Format: [[TN, FP], [FN, TP]]")
    print(f"  {results.confusion_matrix}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = results.confusion_matrix.ravel()
    print(f"\n  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    
    print("\n📊 Classification Report (Test Set):")
    print(results.classification_report)
    
    # Step 5: Save model
    print("\n5. Saving model...")
    try:
        model_path = Path('../models') / 'baseline_banking_model.joblib'
        baseline.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    print("\n✓ Example 2 completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Baseline Model (Logistic Regression) - Usage Examples")
    print("=" * 80)
    print("\nThis script demonstrates:")
    print("  - Training Logistic Regression as interpretable baseline")
    print("  - Evaluation using AUC-PR, F1-Score, and Confusion Matrix")
    print("  - Handling imbalanced data with class weights")
    print("  - Complete workflow for both e-commerce and banking datasets")
    
    # Run examples
    example_ecommerce_baseline()
    example_banking_baseline()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Compare baseline results with ensemble models")
    print("  - Use baseline as interpretable benchmark")
    print("  - Analyze feature importance for business insights")


if __name__ == '__main__':
    main()

