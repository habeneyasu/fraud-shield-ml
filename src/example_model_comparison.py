"""
Example: Using ModelComparator Class

This script demonstrates how to use the ModelComparator class to compare
multiple models side-by-side and select the best model based on performance
metrics and interpretability.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data
from src.data_preparation import DataPreparation
from src.baseline_model import BaselineModel
from src.ensemble_model import EnsembleModel
from src.cross_validation import CrossValidator
from src.model_comparison import ModelComparator, ModelComparisonEntry, ModelComparisonResults


def example_model_comparison():
    """Complete example: Train multiple models, compare, and select best."""
    print("=" * 80)
    print("Model Comparison and Selection - Complete Example")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    try:
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
        print(f"✓ Loaded {len(df)} rows")
        
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
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Train Baseline Model
    print("\n2. Training Baseline Model (Logistic Regression)...")
    try:
        baseline = BaselineModel(
            class_weight='balanced',
            random_state=42
        )
        baseline_results = baseline.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test
        )
        print(f"✓ Baseline model trained")
        print(f"  Test F1-Score: {baseline_results.test_metrics['f1']:.4f}")
        print(f"  Test AUC-PR: {baseline_results.test_metrics['pr_auc']:.4f}")
    except Exception as e:
        print(f"✗ Error training baseline: {e}")
        return
    
    # Step 3: Train Random Forest Model
    print("\n3. Training Random Forest Model...")
    try:
        rf = EnsembleModel(
            model_type='random_forest',
            class_weight='balanced',
            random_state=42
        )
        rf_results = rf.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            param_grid={
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            },
            tune_hyperparameters=True,
            cv=5,
            search_type='grid'
        )
        print(f"✓ Random Forest model trained")
        print(f"  Test F1-Score: {rf_results.test_metrics['f1']:.4f}")
        print(f"  Test AUC-PR: {rf_results.test_metrics['pr_auc']:.4f}")
        print(f"  Best params: {rf_results.best_params}")
    except Exception as e:
        print(f"✗ Error training Random Forest: {e}")
        return
    
    # Step 4: Train XGBoost Model
    print("\n4. Training XGBoost Model...")
    try:
        xgb_model = EnsembleModel(
            model_type='xgboost',
            class_weight='balanced',
            random_state=42
        )
        xgb_results = xgb_model.train_and_evaluate(
            split_result.X_train, split_result.y_train,
            split_result.X_test, split_result.y_test,
            param_grid={
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2]
            },
            tune_hyperparameters=True,
            cv=5,
            search_type='random',
            n_iter=10
        )
        print(f"✓ XGBoost model trained")
        print(f"  Test F1-Score: {xgb_results.test_metrics['f1']:.4f}")
        print(f"  Test AUC-PR: {xgb_results.test_metrics['pr_auc']:.4f}")
        print(f"  Best params: {xgb_results.best_params}")
    except Exception as e:
        print(f"✗ Error training XGBoost: {e}")
        return
    
    # Step 5: Perform Cross-Validation on all models
    print("\n5. Performing Cross-Validation on all models...")
    try:
        cv = CrossValidator(n_folds=5, random_state=42)
        
        # Baseline CV
        print("  Cross-validating baseline model...")
        baseline_cv = cv.cross_validate(
            baseline, split_result.X_train, split_result.y_train
        )
        
        # RF CV
        print("  Cross-validating Random Forest...")
        rf_cv = cv.cross_validate(
            rf, split_result.X_train, split_result.y_train
        )
        
        # XGBoost CV
        print("  Cross-validating XGBoost...")
        xgb_cv = cv.cross_validate(
            xgb_model, split_result.X_train, split_result.y_train
        )
        
        print("✓ Cross-validation completed for all models")
    except Exception as e:
        print(f"✗ Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Create comparison entries
    print("\n6. Creating model comparison entries...")
    try:
        baseline_entry = ModelComparisonEntry(
            model_name='Logistic Regression',
            model_type='baseline',
            test_metrics=baseline_results.test_metrics,
            cv_metrics=baseline_cv.metrics_summary,
            interpretability_score=1.0,  # Highly interpretable
            model_object=baseline.model,
            best_params=None
        )
        
        rf_entry = ModelComparisonEntry(
            model_name='Random Forest',
            model_type='random_forest',
            test_metrics=rf_results.test_metrics,
            cv_metrics=rf_cv.metrics_summary,
            interpretability_score=0.6,  # Feature importance available
            model_object=rf.model,
            best_params=rf_results.best_params
        )
        
        xgb_entry = ModelComparisonEntry(
            model_name='XGBoost',
            model_type='xgboost',
            test_metrics=xgb_results.test_metrics,
            cv_metrics=xgb_cv.metrics_summary,
            interpretability_score=0.5,  # Feature importance available, but complex
            model_object=xgb_model.model,
            best_params=xgb_results.best_params
        )
        
        print("✓ Comparison entries created")
    except Exception as e:
        print(f"✗ Error creating entries: {e}")
        return
    
    # Step 7: Compare models
    print("\n7. Comparing models side-by-side...")
    try:
        # Option 1: Performance-focused (70% performance, 30% interpretability)
        comparator_perf = ModelComparator(
            primary_metric='f1',
            interpretability_weight=0.3,
            performance_weight=0.7,
            consider_cv=True
        )
        
        results_perf = comparator_perf.compare_models([
            baseline_entry, rf_entry, xgb_entry
        ])
        
        print("\n" + "=" * 80)
        print("COMPARISON: Performance-Focused (70% performance, 30% interpretability)")
        print("=" * 80)
        comparator_perf.print_comparison(results_perf)
        
        # Option 2: Balanced (50% performance, 50% interpretability)
        comparator_balanced = ModelComparator(
            primary_metric='f1',
            interpretability_weight=0.5,
            performance_weight=0.5,
            consider_cv=True
        )
        
        results_balanced = comparator_balanced.compare_models([
            baseline_entry, rf_entry, xgb_entry
        ])
        
        print("\n" + "=" * 80)
        print("COMPARISON: Balanced (50% performance, 50% interpretability)")
        print("=" * 80)
        comparator_balanced.print_comparison(results_balanced)
        
        # Option 3: Interpretability-focused (30% performance, 70% interpretability)
        comparator_interp = ModelComparator(
            primary_metric='f1',
            interpretability_weight=0.7,
            performance_weight=0.3,
            consider_cv=True
        )
        
        results_interp = comparator_interp.compare_models([
            baseline_entry, rf_entry, xgb_entry
        ])
        
        print("\n" + "=" * 80)
        print("COMPARISON: Interpretability-Focused (30% performance, 70% interpretability)")
        print("=" * 80)
        comparator_interp.print_comparison(results_interp)
        
    except Exception as e:
        print(f"✗ Error comparing models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 8: Get summary
    print("\n8. Generating comparison summary...")
    try:
        summary = comparator_perf.get_comparison_summary(results_perf)
        
        print("\n📊 Comparison Summary:")
        print("-" * 80)
        print(f"  Best Model: {summary['best_model']}")
        print(f"  Model Type: {summary['best_model_type']}")
        print(f"  Overall Score: {summary['best_overall_score']:.4f}")
        print(f"  Performance Score: {summary['best_performance_score']:.4f}")
        print(f"  Interpretability Score: {summary['best_interpretability_score']:.4f}")
        print(f"  Primary Metric ({summary['primary_metric'].upper()}): {summary['primary_metric_value']:.4f}")
        print(f"  Total Models Compared: {summary['total_models_compared']}")
        
    except Exception as e:
        print(f"✗ Error generating summary: {e}")
    
    print("\n" + "=" * 80)
    print("Model comparison completed successfully!")
    print("=" * 80)
    print("\n💡 Key Insights:")
    print("  - Different weighting schemes may select different best models")
    print("  - Performance-focused: Prioritizes predictive accuracy")
    print("  - Balanced: Balances performance and interpretability")
    print("  - Interpretability-focused: Prioritizes explainability for regulatory compliance")
    print("\n  Choose the weighting scheme that aligns with your business objectives!")


def main():
    """Run the example."""
    print("\n" + "=" * 80)
    print("Model Comparison and Selection - Usage Example")
    print("=" * 80)
    print("\nThis script demonstrates:")
    print("  - Training multiple models (baseline and ensemble)")
    print("  - Performing cross-validation on all models")
    print("  - Side-by-side comparison of models")
    print("  - Selecting best model with clear justification")
    print("  - Considering both performance metrics and interpretability")
    print("  - Different weighting schemes for different business objectives")
    
    example_model_comparison()


if __name__ == '__main__':
    main()

