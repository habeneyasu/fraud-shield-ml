"""
Example: Bivariate Risk Analysis

This script demonstrates how to use the analysis module to perform
targeted bivariate analyses focusing on high-risk patterns with
narrative interpretations for stakeholders.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    load_fraud_data,
    analyze_amount_vs_fraud,
    analyze_device_vs_fraud,
    analyze_source_vs_fraud,
    analyze_browser_vs_fraud,
    generate_risk_summary_report
)


def main():
    """
    Main function demonstrating bivariate risk analysis workflow.
    """
    print("=" * 100)
    print("BIVARIATE RISK ANALYSIS - EXAMPLE WORKFLOW")
    print("=" * 100)
    print("\nThis example demonstrates targeted bivariate analyses focusing on")
    print("high-risk patterns with concise narrative interpretations for stakeholders.\n")
    
    try:
        # Load the fraud dataset
        print("Step 1: Loading fraud e-commerce dataset...")
        data_dir = Path('../data')
        raw_data_path = data_dir / 'raw' / 'Fraud_Data.csv'
        
        df = load_fraud_data(raw_data_path)
        print(f"✓ Dataset loaded successfully! Shape: {df.shape}\n")
        
        # Perform individual bivariate analyses
        print("\n" + "=" * 100)
        print("PERFORMING INDIVIDUAL BIVARIATE ANALYSES")
        print("=" * 100)
        
        # 1. Amount vs Fraud Analysis
        print("\n[1/4] Analyzing Transaction Amount vs Fraud...")
        amount_results = analyze_amount_vs_fraud(
            df,
            amount_column='purchase_value',
            fraud_column='class'
        )
        
        # 2. Device vs Fraud Analysis
        print("\n[2/4] Analyzing Device vs Fraud...")
        device_results = analyze_device_vs_fraud(
            df,
            device_column='device_id',
            fraud_column='class',
            top_n=10
        )
        
        # 3. Source vs Fraud Analysis
        print("\n[3/4] Analyzing Traffic Source vs Fraud...")
        source_results = analyze_source_vs_fraud(
            df,
            source_column='source',
            fraud_column='class'
        )
        
        # 4. Browser vs Fraud Analysis
        print("\n[4/4] Analyzing Browser Type vs Fraud...")
        browser_results = analyze_browser_vs_fraud(
            df,
            browser_column='browser',
            fraud_column='class'
        )
        
        # Generate comprehensive risk summary report
        print("\n" + "=" * 100)
        print("GENERATING COMPREHENSIVE RISK SUMMARY REPORT")
        print("=" * 100)
        
        risk_report = generate_risk_summary_report(
            df,
            amount_column='purchase_value',
            device_column='device_id',
            source_column='source',
            browser_column='browser',
            fraud_column='class'
        )
        
        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE")
        print("=" * 100)
        print("\nAll bivariate analyses have been completed with narrative interpretations.")
        print("Key risk drivers have been identified and prioritized for stakeholders.")
        print("\nNext Steps:")
        print("  1. Review the risk summary report above")
        print("  2. Implement recommended fraud prevention measures")
        print("  3. Monitor high-risk patterns identified in the analysis")
        print("  4. Update fraud detection models with insights from this analysis")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Data file not found: {e}")
        print("Please ensure the fraud dataset is available at: data/raw/Fraud_Data.csv")
        return 1
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

