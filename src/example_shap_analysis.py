"""
Example: SHAP Analysis for Model Explainability

This script demonstrates how to perform SHAP analysis on trained models:
- Generate SHAP Summary Plot (global feature importance)
- Generate SHAP Force Plots for case studies:
  - True Positive (TP): correctly identified fraud
  - False Positive (FP): legitimate transaction flagged as fraud
  - False Negative (FN): missed fraud

This addresses Task 3 - Model Explainability, Instruction 2.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation
from src.ensemble_model import EnsembleModel
from src.model_explainability import ModelExplainability


def example_ecommerce_shap_analysis():
    """
    Example: SHAP analysis for e-commerce fraud detection model.
    """
    print("=" * 80)
    print("SHAP ANALYSIS EXAMPLE - E-COMMERCE DATASET")
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
        
        # Train with basic hyperparameters
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
    
    # Step 4: Initialize explainer
    print("\n4. Initializing ModelExplainability...")
    try:
        explainer = ModelExplainability(
            model=ensemble.model,
            feature_names=split_result.X_train.columns.tolist(),
            random_state=42
        )
        print("✓ Explainer initialized")
    except Exception as e:
        print(f"✗ Error initializing explainer: {e}")
        return
    
    # Step 5: Compute SHAP values
    print("\n5. Computing SHAP values (this may take a moment)...")
    try:
        # Use a sample of test data for efficiency
        shap_values, shap_explainer = explainer.compute_shap_values(
            split_result.X_test,
            sample_size=100,  # Use 100 samples for SHAP computation
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
    
    # Step 6: Generate SHAP Summary Plot
    print("\n6. Generating SHAP Summary Plot (global feature importance)...")
    try:
        summary_path = explainer.generate_shap_summary_plot(
            shap_values,
            split_result.X_test.iloc[:len(shap_values)],  # Match SHAP sample size
            max_display=10,
            save_path='visualizations/shap_summary_ecommerce.png',
            title='SHAP Summary Plot - E-Commerce Fraud Detection\n(Random Forest, Top 10 Features)'
        )
        print(f"✓ SHAP summary plot saved to: {summary_path}")
    except Exception as e:
        print(f"✗ Error generating summary plot: {e}")
        return
    
    # Step 7: Identify case studies (TP, FP, FN)
    print("\n7. Identifying case study instances (TP, FP, FN)...")
    try:
        # Use the same subset for case identification
        X_test_sample = split_result.X_test.iloc[:len(shap_values)]
        y_test_sample = split_result.y_test.iloc[:len(shap_values)]
        
        case_studies = explainer.identify_case_studies(
            X_test_sample,
            y_test_sample,
            min_samples=1
        )
        
        print(f"✓ Identified {len(case_studies)} case study instances:")
        for case in case_studies:
            print(f"  - {case.case_type}: index={case.index}, "
                  f"predicted={case.prediction}, actual={case.actual}, "
                  f"prob={case.probability:.4f}")
        
        if len(case_studies) < 3:
            print("⚠ Warning: Less than 3 case studies found. Some force plots may not be generated.")
    
    except Exception as e:
        print(f"✗ Error identifying case studies: {e}")
        return
    
    # Step 8: Generate SHAP Force Plots for each case type
    print("\n8. Generating SHAP Force Plots for case studies...")
    
    # Group cases by type
    cases_by_type = {}
    for case in case_studies:
        if case.case_type not in cases_by_type:
            cases_by_type[case.case_type] = []
        cases_by_type[case.case_type].append(case)
    
    # Generate force plots
    force_plot_paths = {}
    
    for case_type in ['TP', 'FP', 'FN']:
        if case_type in cases_by_type and len(cases_by_type[case_type]) > 0:
            case = cases_by_type[case_type][0]  # Use first instance of each type
            print(f"\n  Generating force plot for {case_type} (index={case.index})...")
            
            try:
                # Try HTML first (interactive)
                try:
                    force_path = explainer.generate_shap_force_plot(
                        shap_values,
                        X_test_sample,
                        case,
                        save_path=f'visualizations/shap_force_{case_type}_idx{case.index}.html',
                        plot_type='html'
                    )
                    force_plot_paths[case_type] = force_path
                    print(f"    ✓ HTML force plot saved to: {force_path}")
                except Exception:
                    # Fallback to matplotlib
                    force_path = explainer.generate_shap_force_plot(
                        shap_values,
                        X_test_sample,
                        case,
                        save_path=f'visualizations/shap_force_{case_type}_idx{case.index}.png',
                        plot_type='matplotlib'
                    )
                    force_plot_paths[case_type] = force_path
                    print(f"    ✓ Matplotlib force plot saved to: {force_path}")
            
            except Exception as e:
                print(f"    ✗ Error generating force plot for {case_type}: {e}")
        else:
            print(f"  ⚠ No {case_type} cases found. Skipping force plot.")
    
    # Summary
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"✓ SHAP Summary Plot: {summary_path}")
    print(f"✓ Force Plots Generated: {len(force_plot_paths)}")
    for case_type, path in force_plot_paths.items():
        print(f"  - {case_type}: {path}")
    print("\n✓ SHAP analysis completed successfully!")
    print("=" * 80)


def example_banking_shap_analysis():
    """
    Example: SHAP analysis for banking (credit card) fraud detection model.
    """
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS EXAMPLE - BANKING (CREDIT CARD) DATASET")
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
    
    # Step 4: Initialize explainer
    print("\n4. Initializing ModelExplainability...")
    try:
        explainer = ModelExplainability(
            model=ensemble.model,
            feature_names=split_result.X_train.columns.tolist(),
            random_state=42
        )
        print("✓ Explainer initialized")
    except Exception as e:
        print(f"✗ Error initializing explainer: {e}")
        return
    
    # Step 5: Compute SHAP values
    print("\n5. Computing SHAP values (this may take a moment)...")
    try:
        # Use a sample of test data for efficiency
        shap_values, shap_explainer = explainer.compute_shap_values(
            split_result.X_test,
            sample_size=100,  # Use 100 samples for SHAP computation
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
    
    # Step 6: Generate SHAP Summary Plot
    print("\n6. Generating SHAP Summary Plot (global feature importance)...")
    try:
        summary_path = explainer.generate_shap_summary_plot(
            shap_values,
            split_result.X_test.iloc[:len(shap_values)],  # Match SHAP sample size
            max_display=10,
            save_path='visualizations/shap_summary_banking.png',
            title='SHAP Summary Plot - Credit Card Fraud Detection\n(XGBoost, Top 10 Features)'
        )
        print(f"✓ SHAP summary plot saved to: {summary_path}")
    except Exception as e:
        print(f"✗ Error generating summary plot: {e}")
        return
    
    # Step 7: Identify case studies (TP, FP, FN)
    print("\n7. Identifying case study instances (TP, FP, FN)...")
    try:
        # Use the same subset for case identification
        X_test_sample = split_result.X_test.iloc[:len(shap_values)]
        y_test_sample = split_result.y_test.iloc[:len(shap_values)]
        
        case_studies = explainer.identify_case_studies(
            X_test_sample,
            y_test_sample,
            min_samples=1
        )
        
        print(f"✓ Identified {len(case_studies)} case study instances:")
        for case in case_studies:
            print(f"  - {case.case_type}: index={case.index}, "
                  f"predicted={case.prediction}, actual={case.actual}, "
                  f"prob={case.probability:.4f}")
        
        if len(case_studies) < 3:
            print("⚠ Warning: Less than 3 case studies found. Some force plots may not be generated.")
    
    except Exception as e:
        print(f"✗ Error identifying case studies: {e}")
        return
    
    # Step 8: Generate SHAP Force Plots for each case type
    print("\n8. Generating SHAP Force Plots for case studies...")
    
    # Group cases by type
    cases_by_type = {}
    for case in case_studies:
        if case.case_type not in cases_by_type:
            cases_by_type[case.case_type] = []
        cases_by_type[case.case_type].append(case)
    
    # Generate force plots
    force_plot_paths = {}
    
    for case_type in ['TP', 'FP', 'FN']:
        if case_type in cases_by_type and len(cases_by_type[case_type]) > 0:
            case = cases_by_type[case_type][0]  # Use first instance of each type
            print(f"\n  Generating force plot for {case_type} (index={case.index})...")
            
            try:
                # Try HTML first (interactive)
                try:
                    force_path = explainer.generate_shap_force_plot(
                        shap_values,
                        X_test_sample,
                        case,
                        save_path=f'visualizations/shap_force_{case_type}_idx{case.index}.html',
                        plot_type='html'
                    )
                    force_plot_paths[case_type] = force_path
                    print(f"    ✓ HTML force plot saved to: {force_path}")
                except Exception:
                    # Fallback to matplotlib
                    force_path = explainer.generate_shap_force_plot(
                        shap_values,
                        X_test_sample,
                        case,
                        save_path=f'visualizations/shap_force_{case_type}_idx{case.index}.png',
                        plot_type='matplotlib'
                    )
                    force_plot_paths[case_type] = force_path
                    print(f"    ✓ Matplotlib force plot saved to: {force_path}")
            
            except Exception as e:
                print(f"    ✗ Error generating force plot for {case_type}: {e}")
        else:
            print(f"  ⚠ No {case_type} cases found. Skipping force plot.")
    
    # Summary
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"✓ SHAP Summary Plot: {summary_path}")
    print(f"✓ Force Plots Generated: {len(force_plot_paths)}")
    for case_type, path in force_plot_paths.items():
        print(f"  - {case_type}: {path}")
    print("\n✓ SHAP analysis completed successfully!")
    print("=" * 80)


def main():
    """Run all SHAP analysis examples."""
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS FOR MODEL EXPLAINABILITY")
    print("Task 3 - Model Explainability, Instruction 2")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  - SHAP Summary Plot (global feature importance)")
    print("  - SHAP Force Plots for case studies:")
    print("    * True Positive (TP): correctly identified fraud")
    print("    * False Positive (FP): legitimate flagged as fraud")
    print("    * False Negative (FN): missed fraud")
    print("\nNote: SHAP must be installed (pip install shap)")
    print("=" * 80)
    
    # Run e-commerce example
    example_ecommerce_shap_analysis()
    
    # Run banking example
    example_banking_shap_analysis()
    
    print("\n" + "=" * 80)
    print("ALL SHAP ANALYSIS EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Review SHAP visualizations in visualizations/ directory")
    print("  - Analyze case studies for business insights")
    print("  - Proceed to Case Study Analysis (Instruction 3)")


if __name__ == '__main__':
    main()

