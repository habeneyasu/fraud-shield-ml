"""
Generate visualizations for Interim Report 1.

This script creates all required visualizations from the analysis notebooks.
Run this script to generate the visualization images referenced in the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create visualizations directory
vis_dir = Path(__file__).parent.parent / 'visualizations'
vis_dir.mkdir(exist_ok=True)

def generate_class_distribution_comparison():
    """Generate Visualization 1: Class Distribution Comparison"""
    print("Generating Visualization 1: Class Distribution Comparison...")
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data'
    
    # E-commerce data
    df_fraud = pd.read_csv(data_dir / 'raw' / 'Fraud_Data.csv')
    fraud_normal = len(df_fraud[df_fraud['class'] == 0])
    fraud_fraud = len(df_fraud[df_fraud['class'] == 1])
    fraud_total = len(df_fraud)
    
    # Banking data
    df_cc = pd.read_csv(data_dir / 'raw' / 'creditcard.csv')
    cc_normal = len(df_cc[df_cc['Class'] == 0])
    cc_fraud = len(df_cc[df_cc['Class'] == 1])
    cc_total = len(df_cc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # E-commerce
    axes[0].bar(['Normal (0)', 'Fraud (1)'], [fraud_normal, fraud_fraud], 
                color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title(f'E-commerce Dataset\nNormal: {fraud_normal:,} (90.64%)\nFraud: {fraud_fraud:,} (9.36%)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Banking
    axes[1].bar(['Normal (0)', 'Fraud (1)'], [cc_normal, cc_fraud], 
                color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Banking Dataset\nNormal: {cc_normal:,} (99.83%)\nFraud: {cc_fraud:,} (0.17%)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'class_distribution_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: class_distribution_comparison.png")

def generate_purchase_value_vs_fraud():
    """Generate Visualization 2: Purchase Value vs Fraud Label"""
    print("Generating Visualization 2: Purchase Value vs Fraud Label...")
    
    data_dir = Path(__file__).parent.parent / 'data'
    df = pd.read_csv(data_dir / 'raw' / 'Fraud_Data.csv')
    
    fraud_amounts = df[df['class'] == 1]['purchase_value']
    normal_amounts = df[df['class'] == 0]['purchase_value']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    data_to_plot = [normal_amounts, fraud_amounts]
    bp = axes[0].boxplot(data_to_plot, tick_labels=['Normal', 'Fraud'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[0].set_ylabel('Purchase Value ($)', fontsize=12, fontweight='bold')
    axes[0].set_title('Purchase Value Distribution by Fraud Status', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Histogram
    axes[1].hist(normal_amounts, bins=50, alpha=0.6, label='Normal', color='#3498db', edgecolor='black')
    axes[1].hist(fraud_amounts, bins=50, alpha=0.6, label='Fraud', color='#e74c3c', edgecolor='black')
    axes[1].set_xlabel('Purchase Value ($)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Amount Distribution: Normal vs Fraud', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'purchase_value_vs_fraud.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: purchase_value_vs_fraud.png")

def generate_global_fraud_heatmap():
    """Generate Visualization 3: Global Fraud Heatmap"""
    print("Generating Visualization 3: Global Fraud Heatmap...")
    
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Try to load processed data with country
    try:
        df = pd.read_csv(data_dir / 'processed' / 'fraud_data_with_country.csv')
    except FileNotFoundError:
        print("⚠ Processed data with country not found. Generating from raw data...")
        # Load and process
        df = pd.read_csv(data_dir / 'raw' / 'Fraud_Data.csv')
        ip_mapping = pd.read_csv(data_dir / 'raw' / 'IpAddress_to_Country.csv')
        
        # Convert IP to int
        df['ip_address_int'] = df['ip_address'].astype('int64')
        ip_mapping['lower_bound_ip_address'] = ip_mapping['lower_bound_ip_address'].astype('int64')
        ip_mapping['upper_bound_ip_address'] = ip_mapping['upper_bound_ip_address'].astype('int64')
        
        # Merge (simplified - would need proper range lookup in production)
        df = df.merge(ip_mapping[['lower_bound_ip_address', 'country']], 
                     left_on='ip_address_int', right_on='lower_bound_ip_address', how='left')
        df['country'] = df['country'].fillna('Unknown')
    
    # Calculate country fraud stats
    country_stats = df.groupby('country').agg({
        'class': ['count', 'sum']
    }).reset_index()
    country_stats.columns = ['country', 'total_transactions', 'fraud_count']
    country_stats['fraud_rate'] = (country_stats['fraud_count'] / country_stats['total_transactions']) * 100
    
    # Filter countries with at least 100 transactions
    significant = country_stats[country_stats['total_transactions'] >= 100].copy()
    significant = significant.sort_values('fraud_rate', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    colors = ['#c0392b' if x > 20 else '#e74c3c' if x > 15 else '#f39c12' 
              for x in significant['fraud_rate']]
    
    bars = ax.barh(range(len(significant)), significant['fraud_rate'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(significant)))
    ax.set_yticklabels(significant['country'], fontsize=10)
    ax.set_xlabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Countries by Fraud Rate\n(Min 100 transactions)', fontsize=14, fontweight='bold')
    ax.axvline(9.36, color='blue', linestyle='--', linewidth=2, label='Overall Rate: 9.36%')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'global_fraud_heatmap.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: global_fraud_heatmap.png")

def generate_fraud_by_time_since_signup():
    """Generate Visualization 4: Fraud Probability by Time Since Signup"""
    print("Generating Visualization 4: Fraud Probability by Time Since Signup...")
    
    data_dir = Path(__file__).parent.parent / 'data'
    df = pd.read_csv(data_dir / 'raw' / 'Fraud_Data.csv')
    
    # Convert to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    df = df.dropna(subset=['signup_time', 'purchase_time'])
    
    # Calculate time since signup
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Create bins
    bins = [0, 1, 24, 168, float('inf')]
    labels = ['0-1h', '1-24h', '24-168h', '>168h']
    df['time_bin'] = pd.cut(df['time_since_signup'], bins=bins, labels=labels)
    
    # Calculate fraud rate by bin
    fraud_by_bin = df.groupby('time_bin', observed=True).agg({
        'class': ['count', 'sum']
    }).reset_index()
    fraud_by_bin.columns = ['time_bin', 'total', 'fraud_count']
    fraud_by_bin['fraud_rate'] = (fraud_by_bin['fraud_count'] / fraud_by_bin['total']) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(fraud_by_bin['time_bin'], fraud_by_bin['fraud_rate'], 
                  color=['#c0392b', '#e74c3c', '#f39c12', '#3498db'], 
                  alpha=0.7, edgecolor='black')
    ax.axhline(9.36, color='blue', linestyle='--', linewidth=2, label='Baseline: 9.36%')
    ax.set_xlabel('Time Since Signup', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Fraud Rate by Time Since Signup', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, fraud_by_bin['fraud_rate'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'fraud_by_time_since_signup.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: fraud_by_time_since_signup.png")

def generate_feature_correlation_heatmap():
    """Generate Visualization 5: Feature Correlation Heatmap"""
    print("Generating Visualization 5: Feature Correlation Heatmap...")
    
    data_dir = Path(__file__).parent.parent / 'data'
    df = pd.read_csv(data_dir / 'raw' / 'Fraud_Data.csv')
    
    # Select numerical features
    numerical_features = ['purchase_value', 'age', 'ip_address']
    
    # Create correlation matrix
    corr_matrix = df[numerical_features + ['class']].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Heatmap\n(Post Engineering)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'feature_correlation_heatmap.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: feature_correlation_heatmap.png")

def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("Generating Visualizations for Interim Report 1")
    print("=" * 60)
    print()
    
    try:
        generate_class_distribution_comparison()
        generate_purchase_value_vs_fraud()
        generate_global_fraud_heatmap()
        generate_fraud_by_time_since_signup()
        generate_feature_correlation_heatmap()
        
        print()
        print("=" * 60)
        print("✓ All visualizations generated successfully!")
        print(f"✓ Saved to: {vis_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

