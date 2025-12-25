"""
Example: Using EnsembleModel Class

This script demonstrates how to use the EnsembleModel class to train and evaluate
ensemble models (Random Forest, XGBoost, LightGBM) with hyperparameter tuning
for fraud detection on both e-commerce and banking datasets.
"""

from pathlib import Path
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.ensemble_model import EnsembleModel, EnsembleModelResults


def example_random_forest():
    """Example using Random Forest with hyperparameter tuning."""
    print("=" * 80)
    print("Example 1: Random Forest - E-commerce Dataset")
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
    
    # Step 3: Define hyperparameter grid
    print("\n3. Defining hyperparameter grid...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    print(f"  Parameters to tune: {list(param_grid.keys())}")
    print(f"  Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Step 4: Train and evaluate ensemble model
    print("\n4. Training and evaluating Random Forest with hyperparameter tuning...")
    try:
        ensemble = EnsembleModel(
            model_type='random_forest',
            class_weight='balanced',
            random_state=42
        )
        
        results: EnsembleModelResults = ensemble.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            param_grid=param_grid,
            tune_hyperparameters=True,
            cv=5,
            scoring='f1',
            search_type='grid',  # Use 'random' for faster search
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        print("\n✓ Ensemble model training completed")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL RESULTS - RANDOM FOREST")
    print("=" * 80)
    
    print("\n📊 Best Hyperparameters:")
    for param, value in results.best_params.items():
        print(f"  {param}: {value}")
    
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
    
    # Step 6: Feature importance
    print("\n6. Feature importance (Top 10):")
    try:
        importance_df = ensemble.get_feature_importance()
        if importance_df is not None:
            print(importance_df.head(10).to_string(index=False))
        else:
            print("  Feature importance not available")
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")
    
    # Step 7: Save model
    print("\n7. Saving model...")
    try:
        model_path = Path('../models') / 'ensemble_rf_ecommerce_model.joblib'
        ensemble.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    print("\n✓ Example 1 completed successfully!")


def example_xgboost():
    """Example using XGBoost with hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("Example 2: XGBoost - Banking Dataset")
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
    
    # Step 3: Define hyperparameter grid (smaller for faster execution)
    print("\n3. Defining hyperparameter grid...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9]
    }
    print(f"  Parameters to tune: {list(param_grid.keys())}")
    print(f"  Using RandomizedSearchCV for faster search")
    
    # Step 4: Train and evaluate ensemble model
    print("\n4. Training and evaluating XGBoost with hyperparameter tuning...")
    try:
        ensemble = EnsembleModel(
            model_type='xgboost',
            class_weight='balanced',  # Will use scale_pos_weight internally
            random_state=42
        )
        
        results: EnsembleModelResults = ensemble.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            param_grid=param_grid,
            tune_hyperparameters=True,
            cv=5,
            scoring='f1',
            search_type='random',  # Faster than grid search
            n_iter=10,  # Number of random combinations to try
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        print("\n✓ Ensemble model training completed")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL RESULTS - XGBOOST")
    print("=" * 80)
    
    print("\n📊 Best Hyperparameters:")
    for param, value in results.best_params.items():
        print(f"  {param}: {value}")
    
    print("\n📊 Test Set Metrics:")
    print(f"  F1-Score: {results.test_metrics['f1']:.4f}")
    print(f"  AUC-PR: {results.test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC: {results.test_metrics['roc_auc']:.4f}")
    print(f"  Precision: {results.test_metrics['precision']:.4f}")
    print(f"  Recall: {results.test_metrics['recall']:.4f}")
    
    print("\n📊 Confusion Matrix (Test Set):")
    print(f"  {results.confusion_matrix}")
    
    # Step 6: Save model
    print("\n6. Saving model...")
    try:
        model_path = Path('../models') / 'ensemble_xgb_banking_model.joblib'
        ensemble.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    print("\n✓ Example 2 completed successfully!")


def example_lightgbm():
    """Example using LightGBM with hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("Example 3: LightGBM - E-commerce Dataset")
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
    
    # Step 3: Define hyperparameter grid
    print("\n3. Defining hyperparameter grid...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }
    
    # Step 4: Train and evaluate
    print("\n4. Training and evaluating LightGBM...")
    try:
        ensemble = EnsembleModel(
            model_type='lightgbm',
            class_weight='balanced',
            random_state=42
        )
        
        results: EnsembleModelResults = ensemble.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            param_grid=param_grid,
            tune_hyperparameters=True,
            cv=5,
            scoring='f1',
            search_type='random',
            n_iter=10,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        )
        
        print("\n✓ Ensemble model training completed")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL RESULTS - LIGHTGBM")
    print("=" * 80)
    
    print("\n📊 Best Hyperparameters:")
    for param, value in results.best_params.items():
        print(f"  {param}: {value}")
    
    print("\n📊 Test Set Metrics:")
    print(f"  F1-Score: {results.test_metrics['f1']:.4f}")
    print(f"  AUC-PR: {results.test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC: {results.test_metrics['roc_auc']:.4f}")
    
    print("\n✓ Example 3 completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Ensemble Model (Random Forest, XGBoost, LightGBM) - Usage Examples")
    print("=" * 80)
    print("\nThis script demonstrates:")
    print("  - Training ensemble models with hyperparameter tuning")
    print("  - Evaluation using AUC-PR, F1-Score, and Confusion Matrix")
    print("  - GridSearchCV and RandomizedSearchCV for hyperparameter optimization")
    print("  - Handling imbalanced data with class weights")
    print("  - Complete workflow for multiple ensemble algorithms")
    
    # Run examples
    example_random_forest()
    example_xgboost()
    example_lightgbm()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Compare ensemble results with baseline model")
    print("  - Select best model based on business metrics")
    print("  - Use for production deployment")


if __name__ == '__main__':
    main()

