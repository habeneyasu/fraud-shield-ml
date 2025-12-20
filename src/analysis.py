"""
Risk Analysis Module

This module provides targeted bivariate analyses focusing on high-risk patterns
with concise narrative interpretations for stakeholders.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Dict, Tuple
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_amount_vs_fraud(
    df: pd.DataFrame,
    amount_column: str = 'purchase_value',
    fraud_column: str = 'class',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> Dict[str, Union[float, str]]:
    """
    Analyze the relationship between transaction amount and fraud with narrative interpretation.
    
    This analysis identifies high-risk transaction amounts that are more likely to be fraudulent,
    helping stakeholders understand which price ranges require additional scrutiny.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with transaction data
    amount_column : str, default 'purchase_value'
        Column name containing transaction amounts
    fraud_column : str, default 'class'
        Column name containing fraud labels (1=fraud, 0=normal)
    figsize : tuple, default (14, 6)
        Figure size for visualization
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing statistics and narrative interpretation
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        if amount_column not in df.columns:
            raise ValueError(f"Amount column '{amount_column}' not found")
        
        if fraud_column not in df.columns:
            raise ValueError(f"Fraud column '{fraud_column}' not found")
        
        logger.info("Analyzing amount vs fraud relationship")
        
        # Separate fraud and normal transactions
        fraud_amounts = df[df[fraud_column] == 1][amount_column]
        normal_amounts = df[df[fraud_column] == 0][amount_column]
        
        # Calculate statistics
        fraud_mean = fraud_amounts.mean()
        normal_mean = normal_amounts.mean()
        fraud_median = fraud_amounts.median()
        normal_median = normal_amounts.median()
        
        # Statistical test
        try:
            statistic, p_value = stats.mannwhitneyu(fraud_amounts, normal_amounts, alternative='two-sided')
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            statistic, p_value = None, None
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        data_to_plot = [normal_amounts, fraud_amounts]
        bp = axes[0].boxplot(data_to_plot, labels=['Normal', 'Fraud'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[0].set_ylabel('Transaction Amount', fontsize=12, fontweight='bold')
        axes[0].set_title('Transaction Amount Distribution by Fraud Status', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Histogram
        axes[1].hist(normal_amounts, bins=50, alpha=0.6, label='Normal', color='#3498db', edgecolor='black')
        axes[1].hist(fraud_amounts, bins=50, alpha=0.6, label='Fraud', color='#e74c3c', edgecolor='black')
        axes[1].set_xlabel('Transaction Amount', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Amount Distribution: Normal vs Fraud', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")
        
        plt.show()
        
        # Generate narrative interpretation
        risk_ratio = fraud_mean / normal_mean if normal_mean > 0 else 0
        
        if risk_ratio > 1.5:
            risk_level = "HIGH"
            interpretation = (
                f"**CRITICAL FINDING**: Fraudulent transactions show {risk_ratio:.2f}x higher average amounts "
                f"(${fraud_mean:.2f} vs ${normal_mean:.2f}). This suggests fraudsters target higher-value transactions. "
                f"**Recommendation**: Implement additional verification for transactions above ${fraud_median:.2f}."
            )
        elif risk_ratio > 1.2:
            risk_level = "MODERATE"
            interpretation = (
                f"**IMPORTANT**: Fraudulent transactions are {risk_ratio:.2f}x higher on average "
                f"(${fraud_mean:.2f} vs ${normal_mean:.2f}). Moderate risk pattern detected. "
                f"**Recommendation**: Review transactions in the ${fraud_median:.2f} range for enhanced monitoring."
            )
        else:
            risk_level = "LOW"
            interpretation = (
                f"**OBSERVATION**: Amount differences are minimal (${fraud_mean:.2f} vs ${normal_mean:.2f}). "
                f"Amount alone is not a strong fraud indicator. **Recommendation**: Focus on other risk factors."
            )
        
        results = {
            'fraud_mean': float(fraud_mean),
            'normal_mean': float(normal_mean),
            'fraud_median': float(fraud_median),
            'normal_median': float(normal_median),
            'risk_ratio': float(risk_ratio),
            'p_value': float(p_value) if p_value else None,
            'risk_level': risk_level,
            'interpretation': interpretation
        }
        
        logger.info(f"Analysis complete. Risk level: {risk_level}")
        print("\n" + "=" * 100)
        print("AMOUNT vs FRAUD ANALYSIS - KEY INSIGHTS")
        print("=" * 100)
        print(f"\n{interpretation}\n")
        print(f"Statistical Summary:")
        print(f"  Fraud Mean: ${fraud_mean:,.2f}")
        print(f"  Normal Mean: ${normal_mean:,.2f}")
        print(f"  Risk Ratio: {risk_ratio:.2f}x")
        if p_value:
            print(f"  Statistical Significance (p-value): {p_value:.4f}")
        print("=" * 100 + "\n")
        
        return results
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error analyzing amount vs fraud: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def analyze_device_vs_fraud(
    df: pd.DataFrame,
    device_column: str = 'device_id',
    fraud_column: str = 'class',
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> Dict[str, Union[float, str, pd.DataFrame]]:
    """
    Analyze the relationship between device usage and fraud with narrative interpretation.
    
    This analysis identifies devices associated with higher fraud rates, helping stakeholders
    understand device-level risk patterns and implement device-based fraud prevention.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with transaction data
    device_column : str, default 'device_id'
        Column name containing device identifiers
    fraud_column : str, default 'class'
        Column name containing fraud labels (1=fraud, 0=normal)
    top_n : int, default 10
        Number of top devices to analyze
    figsize : tuple, default (14, 6)
        Figure size for visualization
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing statistics and narrative interpretation
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        if device_column not in df.columns:
            raise ValueError(f"Device column '{device_column}' not found")
        
        if fraud_column not in df.columns:
            raise ValueError(f"Fraud column '{fraud_column}' not found")
        
        logger.info("Analyzing device vs fraud relationship")
        
        # Calculate fraud rate by device
        device_stats = df.groupby(device_column).agg({
            fraud_column: ['count', 'sum', 'mean']
        }).reset_index()
        
        device_stats.columns = ['device_id', 'total_transactions', 'fraud_count', 'fraud_rate']
        device_stats = device_stats[device_stats['total_transactions'] >= 5]  # Filter low-volume devices
        device_stats = device_stats.sort_values('fraud_rate', ascending=False)
        
        # Overall fraud rate
        overall_fraud_rate = df[fraud_column].mean()
        
        # Identify high-risk devices
        high_risk_devices = device_stats[device_stats['fraud_rate'] > overall_fraud_rate * 1.5]
        high_risk_devices = high_risk_devices.head(top_n)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Top devices by fraud rate
        top_devices = device_stats.head(top_n)
        colors = ['#e74c3c' if rate > overall_fraud_rate * 1.5 else '#3498db' 
                 for rate in top_devices['fraud_rate']]
        
        axes[0].barh(range(len(top_devices)), top_devices['fraud_rate'], color=colors, alpha=0.7)
        axes[0].axvline(overall_fraud_rate, color='red', linestyle='--', linewidth=2, label=f'Overall Rate: {overall_fraud_rate:.2%}')
        axes[0].set_yticks(range(len(top_devices)))
        axes[0].set_yticklabels([f"Device {i+1}" for i in range(len(top_devices))], fontsize=9)
        axes[0].set_xlabel('Fraud Rate', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Top {top_n} Devices by Fraud Rate', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Device transaction volume vs fraud rate
        scatter = axes[1].scatter(device_stats['total_transactions'], 
                                 device_stats['fraud_rate'],
                                 alpha=0.6, s=50, c=device_stats['fraud_rate'], 
                                 cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
        axes[1].axhline(overall_fraud_rate, color='red', linestyle='--', linewidth=2, 
                       label=f'Overall Rate: {overall_fraud_rate:.2%}')
        axes[1].set_xlabel('Total Transactions', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Fraud Rate', fontsize=12, fontweight='bold')
        axes[1].set_title('Device Transaction Volume vs Fraud Rate', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Fraud Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")
        
        plt.show()
        
        # Generate narrative interpretation
        if len(high_risk_devices) > 0:
            max_fraud_rate = high_risk_devices['fraud_rate'].max()
            max_device = high_risk_devices.loc[high_risk_devices['fraud_rate'].idxmax()]
            
            if max_fraud_rate > overall_fraud_rate * 3:
                risk_level = "CRITICAL"
                interpretation = (
                    f"**CRITICAL FINDING**: {len(high_risk_devices)} devices show fraud rates {max_fraud_rate/overall_fraud_rate:.1f}x "
                    f"higher than average ({max_fraud_rate:.1%} vs {overall_fraud_rate:.1%}). "
                    f"Highest risk device has {max_device['fraud_count']:.0f} fraud cases out of {max_device['total_transactions']:.0f} transactions. "
                    f"**Recommendation**: Immediately flag and investigate these high-risk devices. "
                    f"Consider device-level blocking or enhanced authentication."
                )
            elif max_fraud_rate > overall_fraud_rate * 2:
                risk_level = "HIGH"
                interpretation = (
                    f"**HIGH RISK**: {len(high_risk_devices)} devices show significantly elevated fraud rates "
                    f"({max_fraud_rate:.1%} vs {overall_fraud_rate:.1%}). "
                    f"**Recommendation**: Implement enhanced monitoring for these devices and review their transaction patterns."
                )
            else:
                risk_level = "MODERATE"
                interpretation = (
                    f"**MODERATE RISK**: Some devices show elevated fraud rates, but pattern is not extreme. "
                    f"**Recommendation**: Monitor device-level patterns as part of overall fraud detection strategy."
                )
        else:
            risk_level = "LOW"
            interpretation = (
                f"**OBSERVATION**: No devices show significantly elevated fraud rates. "
                f"Device-level patterns are relatively uniform. "
                f"**Recommendation**: Device ID alone is not a strong fraud indicator. Focus on other risk factors."
            )
        
        results = {
            'overall_fraud_rate': float(overall_fraud_rate),
            'high_risk_device_count': len(high_risk_devices),
            'max_fraud_rate': float(high_risk_devices['fraud_rate'].max()) if len(high_risk_devices) > 0 else 0,
            'risk_level': risk_level,
            'interpretation': interpretation,
            'high_risk_devices': high_risk_devices
        }
        
        logger.info(f"Analysis complete. Risk level: {risk_level}")
        print("\n" + "=" * 100)
        print("DEVICE vs FRAUD ANALYSIS - KEY INSIGHTS")
        print("=" * 100)
        print(f"\n{interpretation}\n")
        print(f"Statistical Summary:")
        print(f"  Overall Fraud Rate: {overall_fraud_rate:.2%}")
        print(f"  High-Risk Devices Identified: {len(high_risk_devices)}")
        if len(high_risk_devices) > 0:
            print(f"  Maximum Device Fraud Rate: {high_risk_devices['fraud_rate'].max():.2%}")
        print("=" * 100 + "\n")
        
        return results
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error analyzing device vs fraud: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def analyze_source_vs_fraud(
    df: pd.DataFrame,
    source_column: str = 'source',
    fraud_column: str = 'class',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Dict[str, Union[float, str, pd.DataFrame]]:
    """
    Analyze the relationship between traffic source and fraud with narrative interpretation.
    
    This analysis identifies which traffic sources (e.g., SEO, Ads, Direct) are associated
    with higher fraud rates, helping stakeholders optimize marketing channels and fraud prevention.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with transaction data
    source_column : str, default 'source'
        Column name containing traffic source
    fraud_column : str, default 'class'
        Column name containing fraud labels (1=fraud, 0=normal)
    figsize : tuple, default (12, 6)
        Figure size for visualization
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing statistics and narrative interpretation
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        if source_column not in df.columns:
            raise ValueError(f"Source column '{source_column}' not found")
        
        if fraud_column not in df.columns:
            raise ValueError(f"Fraud column '{fraud_column}' not found")
        
        logger.info("Analyzing source vs fraud relationship")
        
        # Calculate fraud rate by source
        source_stats = df.groupby(source_column).agg({
            fraud_column: ['count', 'sum', 'mean']
        }).reset_index()
        
        source_stats.columns = ['source', 'total_transactions', 'fraud_count', 'fraud_rate']
        source_stats = source_stats.sort_values('fraud_rate', ascending=False)
        
        overall_fraud_rate = df[fraud_column].mean()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Fraud rate by source
        colors = ['#e74c3c' if rate > overall_fraud_rate * 1.5 else '#3498db' 
                 for rate in source_stats['fraud_rate']]
        
        bars = axes[0].bar(range(len(source_stats)), source_stats['fraud_rate'], 
                          color=colors, alpha=0.7, edgecolor='black')
        axes[0].axhline(overall_fraud_rate, color='red', linestyle='--', linewidth=2, 
                       label=f'Overall Rate: {overall_fraud_rate:.2%}')
        axes[0].set_xticks(range(len(source_stats)))
        axes[0].set_xticklabels(source_stats['source'], rotation=45, ha='right')
        axes[0].set_ylabel('Fraud Rate', fontsize=12, fontweight='bold')
        axes[0].set_title('Fraud Rate by Traffic Source', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Transaction volume by source
        axes[1].bar(range(len(source_stats)), source_stats['total_transactions'], 
                   color='#3498db', alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(len(source_stats)))
        axes[1].set_xticklabels(source_stats['source'], rotation=45, ha='right')
        axes[1].set_ylabel('Total Transactions', fontsize=12, fontweight='bold')
        axes[1].set_title('Transaction Volume by Source', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")
        
        plt.show()
        
        # Generate narrative interpretation
        high_risk_sources = source_stats[source_stats['fraud_rate'] > overall_fraud_rate * 1.5]
        
        if len(high_risk_sources) > 0:
            max_source = source_stats.iloc[0]
            risk_ratio = max_source['fraud_rate'] / overall_fraud_rate
            
            if risk_ratio > 3:
                risk_level = "CRITICAL"
                interpretation = (
                    f"**CRITICAL FINDING**: Source '{max_source['source']}' shows {risk_ratio:.1f}x higher fraud rate "
                    f"({max_source['fraud_rate']:.1%} vs {overall_fraud_rate:.1%}). "
                    f"This source accounts for {max_source['total_transactions']:.0f} transactions with "
                    f"{max_source['fraud_count']:.0f} fraud cases. "
                    f"**Recommendation**: Immediately review and potentially restrict this traffic source. "
                    f"Implement source-specific fraud screening."
                )
            elif risk_ratio > 2:
                risk_level = "HIGH"
                interpretation = (
                    f"**HIGH RISK**: Source '{max_source['source']}' shows significantly elevated fraud rate "
                    f"({max_source['fraud_rate']:.1%} vs {overall_fraud_rate:.1%}). "
                    f"**Recommendation**: Implement enhanced verification for transactions from this source."
                )
            else:
                risk_level = "MODERATE"
                interpretation = (
                    f"**MODERATE RISK**: Some sources show elevated fraud rates. "
                    f"**Recommendation**: Monitor source-level patterns and adjust fraud prevention accordingly."
                )
        else:
            risk_level = "LOW"
            interpretation = (
                f"**OBSERVATION**: Fraud rates are relatively consistent across traffic sources. "
                f"**Recommendation**: Source is not a primary fraud indicator. Focus on other risk factors."
            )
        
        results = {
            'overall_fraud_rate': float(overall_fraud_rate),
            'high_risk_source_count': len(high_risk_sources),
            'max_fraud_rate': float(source_stats['fraud_rate'].max()),
            'risk_level': risk_level,
            'interpretation': interpretation,
            'source_stats': source_stats
        }
        
        logger.info(f"Analysis complete. Risk level: {risk_level}")
        print("\n" + "=" * 100)
        print("SOURCE vs FRAUD ANALYSIS - KEY INSIGHTS")
        print("=" * 100)
        print(f"\n{interpretation}\n")
        print(f"Statistical Summary:")
        print(f"  Overall Fraud Rate: {overall_fraud_rate:.2%}")
        print(f"  High-Risk Sources: {len(high_risk_sources)}")
        print(f"  Maximum Source Fraud Rate: {source_stats['fraud_rate'].max():.2%}")
        print("=" * 100 + "\n")
        
        return results
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error analyzing source vs fraud: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def analyze_browser_vs_fraud(
    df: pd.DataFrame,
    browser_column: str = 'browser',
    fraud_column: str = 'class',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Dict[str, Union[float, str, pd.DataFrame]]:
    """
    Analyze the relationship between browser type and fraud with narrative interpretation.
    
    This analysis identifies browsers associated with higher fraud rates, which can indicate
    bot activity, automated fraud, or compromised accounts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with transaction data
    browser_column : str, default 'browser'
        Column name containing browser information
    fraud_column : str, default 'class'
        Column name containing fraud labels (1=fraud, 0=normal)
    figsize : tuple, default (12, 6)
        Figure size for visualization
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing statistics and narrative interpretation
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        if browser_column not in df.columns:
            raise ValueError(f"Browser column '{browser_column}' not found")
        
        if fraud_column not in df.columns:
            raise ValueError(f"Fraud column '{fraud_column}' not found")
        
        logger.info("Analyzing browser vs fraud relationship")
        
        # Calculate fraud rate by browser
        browser_stats = df.groupby(browser_column).agg({
            fraud_column: ['count', 'sum', 'mean']
        }).reset_index()
        
        browser_stats.columns = ['browser', 'total_transactions', 'fraud_count', 'fraud_rate']
        browser_stats = browser_stats.sort_values('fraud_rate', ascending=False)
        
        overall_fraud_rate = df[fraud_column].mean()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Fraud rate by browser
        colors = ['#e74c3c' if rate > overall_fraud_rate * 1.5 else '#3498db' 
                 for rate in browser_stats['fraud_rate']]
        
        bars = axes[0].bar(range(len(browser_stats)), browser_stats['fraud_rate'], 
                          color=colors, alpha=0.7, edgecolor='black')
        axes[0].axhline(overall_fraud_rate, color='red', linestyle='--', linewidth=2, 
                       label=f'Overall Rate: {overall_fraud_rate:.2%}')
        axes[0].set_xticks(range(len(browser_stats)))
        axes[0].set_xticklabels(browser_stats['browser'], rotation=45, ha='right')
        axes[0].set_ylabel('Fraud Rate', fontsize=12, fontweight='bold')
        axes[0].set_title('Fraud Rate by Browser Type', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Transaction distribution
        axes[1].pie(browser_stats['total_transactions'], labels=browser_stats['browser'], 
                   autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        axes[1].set_title('Transaction Distribution by Browser', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")
        
        plt.show()
        
        # Generate narrative interpretation
        high_risk_browsers = browser_stats[browser_stats['fraud_rate'] > overall_fraud_rate * 1.5]
        
        if len(high_risk_browsers) > 0:
            max_browser = browser_stats.iloc[0]
            risk_ratio = max_browser['fraud_rate'] / overall_fraud_rate
            
            if risk_ratio > 2.5:
                risk_level = "CRITICAL"
                interpretation = (
                    f"**CRITICAL FINDING**: Browser '{max_browser['browser']}' shows {risk_ratio:.1f}x higher fraud rate "
                    f"({max_browser['fraud_rate']:.1%} vs {overall_fraud_rate:.1%}). "
                    f"This may indicate bot activity or automated fraud. "
                    f"**Recommendation**: Implement browser fingerprinting and enhanced verification "
                    f"for transactions from '{max_browser['browser']}'. Consider CAPTCHA or device verification."
                )
            elif risk_ratio > 1.8:
                risk_level = "HIGH"
                interpretation = (
                    f"**HIGH RISK**: Browser '{max_browser['browser']}' shows elevated fraud rate "
                    f"({max_browser['fraud_rate']:.1%} vs {overall_fraud_rate:.1%}). "
                    f"**Recommendation**: Monitor and potentially flag transactions from this browser for review."
                )
            else:
                risk_level = "MODERATE"
                interpretation = (
                    f"**MODERATE RISK**: Some browsers show slightly elevated fraud rates. "
                    f"**Recommendation**: Include browser type as a feature in fraud detection models."
                )
        else:
            risk_level = "LOW"
            interpretation = (
                f"**OBSERVATION**: Fraud rates are relatively consistent across browsers. "
                f"**Recommendation**: Browser type alone is not a strong fraud indicator."
            )
        
        results = {
            'overall_fraud_rate': float(overall_fraud_rate),
            'high_risk_browser_count': len(high_risk_browsers),
            'max_fraud_rate': float(browser_stats['fraud_rate'].max()),
            'risk_level': risk_level,
            'interpretation': interpretation,
            'browser_stats': browser_stats
        }
        
        logger.info(f"Analysis complete. Risk level: {risk_level}")
        print("\n" + "=" * 100)
        print("BROWSER vs FRAUD ANALYSIS - KEY INSIGHTS")
        print("=" * 100)
        print(f"\n{interpretation}\n")
        print(f"Statistical Summary:")
        print(f"  Overall Fraud Rate: {overall_fraud_rate:.2%}")
        print(f"  High-Risk Browsers: {len(high_risk_browsers)}")
        print(f"  Maximum Browser Fraud Rate: {browser_stats['fraud_rate'].max():.2%}")
        print("=" * 100 + "\n")
        
        return results
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error analyzing browser vs fraud: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def generate_risk_summary_report(
    df: pd.DataFrame,
    amount_column: str = 'purchase_value',
    device_column: str = 'device_id',
    source_column: str = 'source',
    browser_column: str = 'browser',
    fraud_column: str = 'class'
) -> str:
    """
    Generate a comprehensive risk summary report combining all bivariate analyses.
    
    This function provides stakeholders with a consolidated view of key risk drivers
    and actionable recommendations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with transaction data
    amount_column : str, default 'purchase_value'
        Column name for transaction amounts
    device_column : str, default 'device_id'
        Column name for device identifiers
    source_column : str, default 'source'
        Column name for traffic source
    browser_column : str, default 'browser'
        Column name for browser type
    fraud_column : str, default 'class'
        Column name for fraud labels
    
    Returns:
    --------
    str
        Comprehensive risk summary report
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        logger.info("Generating comprehensive risk summary report")
        
        # Run all analyses
        amount_results = analyze_amount_vs_fraud(df, amount_column, fraud_column)
        device_results = analyze_device_vs_fraud(df, device_column, fraud_column)
        source_results = analyze_source_vs_fraud(df, source_column, fraud_column)
        browser_results = analyze_browser_vs_fraud(df, browser_column, fraud_column)
        
        # Compile summary report
        report = f"""
{'='*100}
FRAUD RISK ANALYSIS - EXECUTIVE SUMMARY
{'='*100}

OVERVIEW:
This report analyzes key risk drivers for fraudulent transactions based on bivariate analysis
of transaction patterns. The analysis identifies high-risk patterns that require immediate
attention and provides actionable recommendations for fraud prevention.

KEY FINDINGS:

1. TRANSACTION AMOUNT RISK
   Risk Level: {amount_results['risk_level']}
   {amount_results['interpretation']}

2. DEVICE-LEVEL RISK
   Risk Level: {device_results['risk_level']}
   {device_results['interpretation']}

3. TRAFFIC SOURCE RISK
   Risk Level: {source_results['risk_level']}
   {source_results['interpretation']}

4. BROWSER-TYPE RISK
   Risk Level: {browser_results['risk_level']}
   {browser_results['interpretation']}

PRIORITY ACTIONS:
"""
        
        # Add priority actions based on risk levels
        critical_findings = []
        if amount_results['risk_level'] in ['CRITICAL', 'HIGH']:
            critical_findings.append("• Implement amount-based fraud screening thresholds")
        if device_results['risk_level'] in ['CRITICAL', 'HIGH']:
            critical_findings.append("• Flag and investigate high-risk devices")
        if source_results['risk_level'] in ['CRITICAL', 'HIGH']:
            critical_findings.append("• Review and restrict high-risk traffic sources")
        if browser_results['risk_level'] in ['CRITICAL', 'HIGH']:
            critical_findings.append("• Enhance verification for high-risk browsers")
        
        if critical_findings:
            report += "\n".join(critical_findings)
        else:
            report += "• Continue monitoring all risk factors as part of comprehensive fraud detection strategy"
        
        report += f"""

STATISTICAL SUMMARY:
  Overall Fraud Rate: {df[fraud_column].mean():.2%}
  Total Transactions Analyzed: {len(df):,}
  Fraud Cases: {df[fraud_column].sum():,}
  Normal Cases: {(df[fraud_column] == 0).sum():,}

{'='*100}
"""
        
        logger.info("Risk summary report generated successfully")
        print(report)
        
        return report
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error generating risk summary report: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

