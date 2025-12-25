"""
Comprehensive Model Comparison Script

This script trains multiple models (baseline and ensemble), performs cross-validation,
compares them side-by-side, and generates a detailed report documenting which model
is selected and why, considering both performance and interpretability.

Usage:
    python scripts/comprehensive_model_comparison.py --dataset ecommerce
    python scripts/comprehensive_model_comparison.py --dataset banking
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.baseline_model import BaselineModel
from src.ensemble_model import EnsembleModel
from src.cross_validation import CrossValidator
from src.model_comparison import ModelComparator, ModelComparisonEntry


class ModelComparisonReport:
    """Generate comprehensive model comparison reports."""
    
    def __init__(self, output_dir: Path = Path('reports')):
        """
        Initialize report generator.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def generate_report(
        self,
        dataset_name: str,
        comparison_results,
        all_entries: list,
        cv_results_dict: dict
    ) -> Path:
        """
        Generate comprehensive comparison report.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        comparison_results : ModelComparisonResults
            Results from model comparison
        all_entries : list
            List of ModelComparisonEntry objects
        cv_results_dict : dict
            Dictionary mapping model names to CV results
        
        Returns:
        --------
        Path
            Path to generated report file
        """
        report_path = self.output_dir / f'model_comparison_{dataset_name}_{self.timestamp}.txt'
        
        with open(report_path, 'w') as f:
            # Header
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Dataset: {dataset_name.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total Models Compared: {len(all_entries)}\n")
            f.write(f"Best Model Selected: {comparison_results.best_model_name}\n")
            f.write(f"Selection Criteria: Performance ({comparison_results.comparison_df.loc[comparison_results.comparison_df['model_name'] == comparison_results.best_model_name, 'performance_score'].iloc[0]:.4f}) "
                   f"and Interpretability ({comparison_results.comparison_df.loc[comparison_results.comparison_df['model_name'] == comparison_results.best_model_name, 'interpretability_score'].iloc[0]:.4f})\n")
            f.write("\n")
            
            # Detailed Justification
            f.write("MODEL SELECTION JUSTIFICATION\n")
            f.write("-" * 100 + "\n")
            f.write(comparison_results.best_model_justification + "\n")
            f.write("\n\n")
            
            # Side-by-Side Comparison Table
            f.write("SIDE-BY-SIDE MODEL COMPARISON\n")
            f.write("-" * 100 + "\n")
            df = comparison_results.comparison_df.copy()
            
            # Format for display
            display_cols = ['model_name', 'model_type', 'interpretability_score']
            test_cols = [col for col in df.columns if col.startswith('test_')]
            cv_mean_cols = [col for col in df.columns if col.startswith('cv_') and col.endswith('_mean')]
            display_cols.extend(sorted(test_cols))
            display_cols.extend(sorted(cv_mean_cols))
            display_cols.extend(['performance_score', 'overall_score'])
            
            display_cols = [col for col in display_cols if col in df.columns]
            display_df = df[display_cols].copy()
            
            # Round numeric columns
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            display_df[numeric_cols] = display_df[numeric_cols].round(4)
            
            f.write(display_df.to_string(index=False) + "\n")
            f.write("\n\n")
            
            # Cross-Validation Details
            f.write("CROSS-VALIDATION RESULTS (Mean ± Std)\n")
            f.write("-" * 100 + "\n")
            for model_name, cv_result in cv_results_dict.items():
                f.write(f"\n{model_name}:\n")
                for metric, stats in cv_result.metrics_summary.items():
                    if stats['mean'] is not None:
                        f.write(f"  {metric.upper():12s}: {stats['mean']:7.4f} (± {stats['std']:7.4f}) "
                               f"[Range: {stats['min']:.4f} - {stats['max']:.4f}]\n")
            f.write("\n\n")
            
            # Model Ranking
            f.write("MODEL RANKING (by Overall Score)\n")
            f.write("-" * 100 + "\n")
            for rank, model_name in enumerate(comparison_results.ranking, 1):
                row = df[df['model_name'] == model_name].iloc[0]
                f.write(f"{rank}. {model_name}\n")
                f.write(f"   Overall Score: {row['overall_score']:.4f}\n")
                f.write(f"   Performance Score: {row['performance_score']:.4f}\n")
                f.write(f"   Interpretability Score: {row['interpretability_score']:.4f}\n")
                if f'test_{comparison_results.comparison_df.columns[comparison_results.comparison_df.columns.str.startswith("test_")][0].replace("test_", "")}' in row:
                    primary_metric = 'f1'  # Default
                    if f'test_{primary_metric}' in row:
                        f.write(f"   Test {primary_metric.upper()}: {row[f'test_{primary_metric}']:.4f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 100 + "\n")
            best_entry = next(e for e in all_entries if e.model_name == comparison_results.best_model_name)
            
            f.write(f"1. Primary Recommendation: Deploy {comparison_results.best_model_name}\n")
            f.write(f"   - Best balance of performance and interpretability\n")
            f.write(f"   - Overall score: {df[df['model_name'] == comparison_results.best_model_name]['overall_score'].iloc[0]:.4f}\n")
            f.write("\n")
            
            # Alternative recommendations
            if len(comparison_results.ranking) > 1:
                second_best = comparison_results.ranking[1]
                f.write(f"2. Alternative: Consider {second_best} if interpretability is less critical\n")
                f.write(f"   - May offer better performance for specific use cases\n")
                f.write("\n")
            
            f.write("3. Next Steps:\n")
            f.write("   - Validate selected model on holdout test set\n")
            f.write("   - Implement model monitoring for production deployment\n")
            f.write("   - Set up SHAP explainability for model interpretability\n")
            f.write("   - Establish retraining schedule based on performance drift\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")
        
        print(f"✓ Report saved to: {report_path}")
        return report_path
    
    def save_json_summary(
        self,
        dataset_name: str,
        comparison_results,
        cv_results_dict: dict
    ) -> Path:
        """
        Save comparison results as JSON for programmatic access.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        comparison_results : ModelComparisonResults
            Results from model comparison
        cv_results_dict : dict
            Dictionary mapping model names to CV results
        
        Returns:
        --------
        Path
            Path to JSON file
        """
        json_path = self.output_dir / f'model_comparison_{dataset_name}_{self.timestamp}.json'
        
        summary = {
            'dataset': dataset_name,
            'timestamp': self.timestamp,
            'best_model': {
                'name': comparison_results.best_model_name,
                'justification': comparison_results.best_model_justification
            },
            'ranking': comparison_results.ranking,
            'comparison_table': comparison_results.comparison_df.to_dict('records'),
            'cross_validation': {
                name: {
                    'metrics_summary': result.metrics_summary,
                    'n_folds': result.n_folds
                }
                for name, result in cv_results_dict.items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ JSON summary saved to: {json_path}")
        return json_path


def train_and_compare_models(dataset_type: str = 'ecommerce', verbose: bool = True):
    """
    Train multiple models, perform cross-validation, and compare them.
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset: 'ecommerce' or 'banking'
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    tuple
        (comparison_results, all_entries, cv_results_dict)
    """
    if verbose:
        print("=" * 100)
        print(f"COMPREHENSIVE MODEL COMPARISON - {dataset_type.upper()} DATASET")
        print("=" * 100)
    
    # Step 1: Load and prepare data
    if verbose:
        print("\n[1/6] Loading and preparing data...")
    
    if dataset_type == 'ecommerce':
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
    else:
        df = load_creditcard_data(data_dir='../data', filename='creditcard.csv')
    
    prep = DataPreparation(
        dataset_type=dataset_type,
        test_size=0.2,
        random_state=42,
        stratify=True
    )
    split_result = prep.prepare_and_split(df)
    
    if verbose:
        print(f"✓ Data loaded: {len(df)} rows")
        print(f"✓ Train set: {split_result.train_size} samples")
        print(f"✓ Test set: {split_result.test_size} samples")
    
    # Step 2: Train Baseline Model
    if verbose:
        print("\n[2/6] Training Baseline Model (Logistic Regression)...")
    
    baseline = BaselineModel(class_weight='balanced', random_state=42)
    baseline_results = baseline.train_and_evaluate(
        split_result.X_train, split_result.y_train,
        split_result.X_test, split_result.y_test
    )
    
    if verbose:
        print(f"✓ Baseline trained - Test F1: {baseline_results.test_metrics['f1']:.4f}")
    
    # Step 3: Train Ensemble Models
    if verbose:
        print("\n[3/6] Training Ensemble Models...")
    
    ensemble_models = {}
    ensemble_results = {}
    
    # Random Forest
    if verbose:
        print("  Training Random Forest...")
    rf = EnsembleModel(model_type='random_forest', class_weight='balanced', random_state=42)
    rf_results = rf.train_and_evaluate(
        split_result.X_train, split_result.y_train,
        split_result.X_test, split_result.y_test,
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        tune_hyperparameters=True,
        cv=5,
        search_type='grid'
    )
    ensemble_models['Random Forest'] = rf
    ensemble_results['Random Forest'] = rf_results
    if verbose:
        print(f"    ✓ RF trained - Test F1: {rf_results.test_metrics['f1']:.4f}")
    
    # XGBoost
    if verbose:
        print("  Training XGBoost...")
    xgb_model = EnsembleModel(model_type='xgboost', class_weight='balanced', random_state=42)
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
    ensemble_models['XGBoost'] = xgb_model
    ensemble_results['XGBoost'] = xgb_results
    if verbose:
        print(f"    ✓ XGBoost trained - Test F1: {xgb_results.test_metrics['f1']:.4f}")
    
    # LightGBM
    if verbose:
        print("  Training LightGBM...")
    lgb_model = EnsembleModel(model_type='lightgbm', class_weight='balanced', random_state=42)
    lgb_results = lgb_model.train_and_evaluate(
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
    ensemble_models['LightGBM'] = lgb_model
    ensemble_results['LightGBM'] = lgb_results
    if verbose:
        print(f"    ✓ LightGBM trained - Test F1: {lgb_results.test_metrics['f1']:.4f}")
    
    # Step 4: Cross-Validation on all models
    if verbose:
        print("\n[4/6] Performing Cross-Validation on all models...")
    
    cv = CrossValidator(n_folds=5, random_state=42)
    cv_results_dict = {}
    
    # Baseline CV
    if verbose:
        print("  Cross-validating Baseline...")
    baseline_cv = cv.cross_validate(baseline, split_result.X_train, split_result.y_train)
    cv_results_dict['Logistic Regression'] = baseline_cv
    
    # Ensemble CV
    for name, model in ensemble_models.items():
        if verbose:
            print(f"  Cross-validating {name}...")
        model_cv = cv.cross_validate(model, split_result.X_train, split_result.y_train)
        cv_results_dict[name] = model_cv
    
    if verbose:
        print("✓ Cross-validation completed for all models")
    
    # Step 5: Create comparison entries
    if verbose:
        print("\n[5/6] Creating model comparison entries...")
    
    all_entries = []
    
    # Baseline entry
    baseline_entry = ModelComparisonEntry(
        model_name='Logistic Regression',
        model_type='baseline',
        test_metrics=baseline_results.test_metrics,
        cv_metrics=baseline_cv.metrics_summary,
        interpretability_score=1.0,
        model_object=baseline.model,
        best_params=None
    )
    all_entries.append(baseline_entry)
    
    # Ensemble entries
    interpretability_scores = {
        'Random Forest': 0.6,
        'XGBoost': 0.5,
        'LightGBM': 0.5
    }
    
    for name, results in ensemble_results.items():
        entry = ModelComparisonEntry(
            model_name=name,
            model_type=name.lower().replace(' ', '_'),
            test_metrics=results.test_metrics,
            cv_metrics=cv_results_dict[name].metrics_summary,
            interpretability_score=interpretability_scores[name],
            model_object=ensemble_models[name].model,
            best_params=results.best_params
        )
        all_entries.append(entry)
    
    if verbose:
        print(f"✓ Created {len(all_entries)} comparison entries")
    
    # Step 6: Compare models
    if verbose:
        print("\n[6/6] Comparing models and selecting best...")
    
    # Performance-focused comparison
    comparator = ModelComparator(
        primary_metric='f1',
        interpretability_weight=0.3,
        performance_weight=0.7,
        consider_cv=True
    )
    
    comparison_results = comparator.compare_models(all_entries)
    
    if verbose:
        print(f"✓ Best model selected: {comparison_results.best_model_name}")
        comparator.print_comparison(comparison_results)
    
    return comparison_results, all_entries, cv_results_dict


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive model comparison for fraud detection'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ecommerce', 'banking'],
        default='ecommerce',
        help='Dataset to use for comparison'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save reports'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    try:
        # Train and compare models
        comparison_results, all_entries, cv_results_dict = train_and_compare_models(
            dataset_type=args.dataset,
            verbose=not args.quiet
        )
        
        # Generate reports
        report_generator = ModelComparisonReport(output_dir=Path(args.output_dir))
        
        print("\n" + "=" * 100)
        print("GENERATING REPORTS")
        print("=" * 100)
        
        # Text report
        report_path = report_generator.generate_report(
            args.dataset,
            comparison_results,
            all_entries,
            cv_results_dict
        )
        
        # JSON summary
        json_path = report_generator.save_json_summary(
            args.dataset,
            comparison_results,
            cv_results_dict
        )
        
        print("\n" + "=" * 100)
        print("COMPARISON COMPLETE")
        print("=" * 100)
        print(f"Best Model: {comparison_results.best_model_name}")
        print(f"Report: {report_path}")
        print(f"JSON Summary: {json_path}")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

