"""
Model Comparison and Selection Module

This module provides a professional, reusable class-based approach for comparing
multiple models side-by-side and selecting the best model based on performance
metrics and interpretability considerations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonEntry:
    """
    Data class to hold information about a single model for comparison.
    
    Attributes:
    -----------
    model_name : str
        Name/identifier of the model
    model_type : str
        Type of model: 'baseline', 'random_forest', 'xgboost', 'lightgbm'
    test_metrics : Dict[str, float]
        Test set metrics (e.g., {'f1': 0.85, 'pr_auc': 0.82, ...})
    cv_metrics : Optional[Dict[str, Dict[str, float]]]
        Cross-validation metrics with mean and std
        Format: {'f1': {'mean': 0.85, 'std': 0.02}, ...}
    interpretability_score : float
        Interpretability score (0-1, higher is more interpretable)
    model_object : Any, optional
        The actual model object
    best_params : Optional[Dict[str, Any]]
        Best hyperparameters (for ensemble models)
    """
    model_name: str
    model_type: str
    test_metrics: Dict[str, float]
    cv_metrics: Optional[Dict[str, Dict[str, float]]] = None
    interpretability_score: float = 0.5
    model_object: Any = None
    best_params: Optional[Dict[str, Any]] = None


@dataclass
class ModelComparisonResults:
    """
    Data class to hold model comparison results.
    
    Attributes:
    -----------
    comparison_df : pd.DataFrame
        Side-by-side comparison DataFrame
    best_model_name : str
        Name of the best model
    best_model_justification : str
        Clear justification for why this model was selected
    ranking : List[str]
        List of model names ranked by overall score
    """
    comparison_df: pd.DataFrame
    best_model_name: str
    best_model_justification: str
    ranking: List[str]


class ModelComparator:
    """
    Professional model comparison and selection class for fraud detection.
    
    This class provides a reusable, object-oriented approach to:
    - Compare multiple models side-by-side
    - Select the best model with clear justification
    - Consider both performance metrics and interpretability
    - Generate comprehensive comparison reports
    
    Attributes:
    -----------
    primary_metric : str
        Primary metric for model selection (default: 'f1')
    interpretability_weight : float
        Weight for interpretability in overall scoring (0-1)
    performance_weight : float
        Weight for performance in overall scoring (0-1)
    
    Example:
    --------
    >>> from src.model_comparison import ModelComparator, ModelComparisonEntry
    >>> from src.baseline_model import BaselineModelResults
    >>> from src.ensemble_model import EnsembleModelResults
    >>> 
    >>> # Create comparison entries
    >>> baseline_entry = ModelComparisonEntry(
    ...     model_name='Logistic Regression',
    ...     model_type='baseline',
    ...     test_metrics=baseline_results.test_metrics,
    ...     interpretability_score=1.0
    ... )
    >>> 
    >>> rf_entry = ModelComparisonEntry(
    ...     model_name='Random Forest',
    ...     model_type='random_forest',
    ...     test_metrics=rf_results.test_metrics,
    ...     interpretability_score=0.6
    ... )
    >>> 
    >>> # Compare models
    >>> comparator = ModelComparator(
    ...     primary_metric='f1',
    ...     interpretability_weight=0.3,
    ...     performance_weight=0.7
    ... )
    >>> 
    >>> results = comparator.compare_models([baseline_entry, rf_entry])
    >>> print(f"Best model: {results.best_model_name}")
    >>> print(f"Justification: {results.best_model_justification}")
    """
    
    # Interpretability scores for different model types
    INTERPRETABILITY_SCORES = {
        'baseline': 1.0,  # Logistic Regression - highly interpretable
        'random_forest': 0.6,  # Feature importance available
        'xgboost': 0.5,  # Feature importance available, but complex
        'lightgbm': 0.5,  # Feature importance available, but complex
    }
    
    # Default primary metrics for fraud detection
    PRIMARY_METRICS = ['f1', 'pr_auc', 'roc_auc']
    
    def __init__(
        self,
        primary_metric: str = 'f1',
        interpretability_weight: float = 0.3,
        performance_weight: float = 0.7,
        consider_cv: bool = True
    ):
        """
        Initialize ModelComparator instance.
        
        Parameters:
        -----------
        primary_metric : str, default 'f1'
            Primary metric for model selection. Options: 'f1', 'pr_auc', 'roc_auc'
        interpretability_weight : float, default 0.3
            Weight for interpretability in overall scoring (0-1)
            Higher value means interpretability is more important
        performance_weight : float, default 0.7
            Weight for performance in overall scoring (0-1)
            Higher value means performance is more important
        consider_cv : bool, default True
            Whether to consider cross-validation metrics if available
        
        Raises:
        -------
        ValueError
            If weights don't sum to 1.0 or primary_metric is invalid
        """
        if primary_metric not in self.PRIMARY_METRICS:
            raise ValueError(
                f"Invalid primary_metric: {primary_metric}. "
                f"Must be one of: {self.PRIMARY_METRICS}"
            )
        
        if abs(interpretability_weight + performance_weight - 1.0) > 1e-6:
            raise ValueError(
                f"interpretability_weight ({interpretability_weight}) + "
                f"performance_weight ({performance_weight}) must equal 1.0"
            )
        
        self.primary_metric = primary_metric
        self.interpretability_weight = interpretability_weight
        self.performance_weight = performance_weight
        self.consider_cv = consider_cv
        
        logger.info("Initialized ModelComparator")
        logger.info(f"  Primary metric: {primary_metric}")
        logger.info(f"  Interpretability weight: {interpretability_weight}")
        logger.info(f"  Performance weight: {performance_weight}")
    
    def _normalize_metric(self, value: float, max_value: float, min_value: float) -> float:
        """
        Normalize a metric value to 0-1 scale.
        
        Parameters:
        -----------
        value : float
            Metric value to normalize
        max_value : float
            Maximum value in the dataset
        min_value : float
            Minimum value in the dataset
        
        Returns:
        --------
        float
            Normalized value (0-1)
        """
        if max_value == min_value:
            return 1.0
        return (value - min_value) / (max_value - min_value)
    
    def _calculate_performance_score(
        self,
        entry: ModelComparisonEntry
    ) -> float:
        """
        Calculate performance score for a model entry.
        
        Parameters:
        -----------
        entry : ModelComparisonEntry
            Model entry to score
        
        Returns:
        --------
        float
            Performance score (0-1, higher is better)
        """
        # Use CV metrics if available and requested, otherwise use test metrics
        if self.consider_cv and entry.cv_metrics is not None:
            # Use mean - std for conservative estimate
            primary_value = entry.cv_metrics.get(self.primary_metric, {})
            if primary_value:
                score = primary_value.get('mean', 0.0) - primary_value.get('std', 0.0)
            else:
                # Fallback to test metrics
                score = entry.test_metrics.get(self.primary_metric, 0.0)
        else:
            score = entry.test_metrics.get(self.primary_metric, 0.0)
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _calculate_overall_score(
        self,
        entry: ModelComparisonEntry,
        max_performance: float,
        min_performance: float
    ) -> float:
        """
        Calculate overall score combining performance and interpretability.
        
        Parameters:
        -----------
        entry : ModelComparisonEntry
            Model entry to score
        max_performance : float
            Maximum performance score across all models
        min_performance : float
            Minimum performance score across all models
        
        Returns:
        --------
        float
            Overall score (0-1, higher is better)
        """
        # Normalize performance score
        performance_score = self._calculate_performance_score(entry)
        normalized_performance = self._normalize_metric(
            performance_score, max_performance, min_performance
        )
        
        # Get interpretability score
        interpretability = entry.interpretability_score
        
        # Weighted combination
        overall_score = (
            self.performance_weight * normalized_performance +
            self.interpretability_weight * interpretability
        )
        
        return overall_score
    
    def compare_models(
        self,
        model_entries: List[ModelComparisonEntry]
    ) -> ModelComparisonResults:
        """
        Compare multiple models side-by-side and select the best model.
        
        Parameters:
        -----------
        model_entries : List[ModelComparisonEntry]
            List of model entries to compare
        
        Returns:
        --------
        ModelComparisonResults
            Dataclass containing comparison results and best model selection
        
        Raises:
        -------
        ValueError
            If model_entries is empty or invalid
        """
        if not model_entries:
            raise ValueError("model_entries cannot be empty")
        
        if len(model_entries) < 2:
            logger.warning("Only one model provided for comparison")
        
        logger.info(f"Comparing {len(model_entries)} models")
        
        # Extract all metrics present in any model
        all_metrics = set()
        for entry in model_entries:
            all_metrics.update(entry.test_metrics.keys())
            if entry.cv_metrics:
                all_metrics.update(entry.cv_metrics.keys())
        
        # Create comparison DataFrame
        comparison_data = []
        
        for entry in model_entries:
            row = {
                'model_name': entry.model_name,
                'model_type': entry.model_type,
                'interpretability_score': entry.interpretability_score
            }
            
            # Add test metrics
            for metric in all_metrics:
                if metric in entry.test_metrics:
                    row[f'test_{metric}'] = entry.test_metrics[metric]
                else:
                    row[f'test_{metric}'] = None
            
            # Add CV metrics if available
            if entry.cv_metrics:
                for metric in all_metrics:
                    if metric in entry.cv_metrics:
                        cv_mean = entry.cv_metrics[metric].get('mean')
                        cv_std = entry.cv_metrics[metric].get('std')
                        row[f'cv_{metric}_mean'] = cv_mean
                        row[f'cv_{metric}_std'] = cv_std
                    else:
                        row[f'cv_{metric}_mean'] = None
                        row[f'cv_{metric}_std'] = None
            
            # Add best params if available
            if entry.best_params:
                row['best_params'] = str(entry.best_params)
            else:
                row['best_params'] = None
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate performance scores
        performance_scores = [
            self._calculate_performance_score(entry)
            for entry in model_entries
        ]
        max_performance = max(performance_scores) if performance_scores else 1.0
        min_performance = min(performance_scores) if performance_scores else 0.0
        
        # Calculate overall scores
        overall_scores = [
            self._calculate_overall_score(entry, max_performance, min_performance)
            for entry in model_entries
        ]
        
        # Add scores to DataFrame
        comparison_df['performance_score'] = performance_scores
        comparison_df['overall_score'] = overall_scores
        
        # Rank models by overall score
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)
        ranking = comparison_df['model_name'].tolist()
        
        # Select best model
        best_model_name = ranking[0]
        best_entry = next(
            entry for entry in model_entries
            if entry.model_name == best_model_name
        )
        
        # Generate justification
        justification = self._generate_justification(
            best_entry, model_entries, comparison_df
        )
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"  Overall score: {comparison_df.loc[comparison_df['model_name'] == best_model_name, 'overall_score'].iloc[0]:.4f}")
        
        results = ModelComparisonResults(
            comparison_df=comparison_df,
            best_model_name=best_model_name,
            best_model_justification=justification,
            ranking=ranking
        )
        
        return results
    
    def _generate_justification(
        self,
        best_entry: ModelComparisonEntry,
        all_entries: List[ModelComparisonEntry],
        comparison_df: pd.DataFrame
    ) -> str:
        """
        Generate clear justification for model selection.
        
        Parameters:
        -----------
        best_entry : ModelComparisonEntry
            The selected best model entry
        all_entries : List[ModelComparisonEntry]
            All model entries
        comparison_df : pd.DataFrame
            Comparison DataFrame
        
        Returns:
        --------
        str
            Justification text
        """
        best_row = comparison_df[comparison_df['model_name'] == best_entry.model_name].iloc[0]
        
        justification_parts = []
        
        # Primary metric performance
        if self.consider_cv and best_entry.cv_metrics and self.primary_metric in best_entry.cv_metrics:
            cv_metric = best_entry.cv_metrics[self.primary_metric]
            justification_parts.append(
                f"{best_entry.model_name} achieved the best overall score "
                f"({best_row['overall_score']:.4f}) with a {self.primary_metric.upper()} of "
                f"{cv_metric['mean']:.4f} (±{cv_metric['std']:.4f}) from cross-validation."
            )
        else:
            test_value = best_entry.test_metrics.get(self.primary_metric, 0.0)
            justification_parts.append(
                f"{best_entry.model_name} achieved the best overall score "
                f"({best_row['overall_score']:.4f}) with a {self.primary_metric.upper()} of "
                f"{test_value:.4f} on the test set."
            )
        
        # Performance comparison
        other_models = [e for e in all_entries if e.model_name != best_entry.model_name]
        if other_models:
            best_primary = best_entry.test_metrics.get(self.primary_metric, 0.0)
            comparisons = []
            for other in other_models:
                other_primary = other.test_metrics.get(self.primary_metric, 0.0)
                if best_primary > other_primary:
                    diff = best_primary - other_primary
                    comparisons.append(
                        f"{diff:.4f} higher than {other.model_name}"
                    )
            
            if comparisons:
                justification_parts.append(
                    f"Performance: {self.primary_metric.upper()} is "
                    f"{', '.join(comparisons)}."
                )
        
        # Interpretability note
        if best_entry.interpretability_score >= 0.7:
            justification_parts.append(
                f"Interpretability: High interpretability score "
                f"({best_entry.interpretability_score:.2f}) makes this model "
                f"suitable for regulatory compliance and stakeholder communication."
            )
        elif best_entry.interpretability_score >= 0.4:
            justification_parts.append(
                f"Interpretability: Moderate interpretability score "
                f"({best_entry.interpretability_score:.2f}). "
                f"Feature importance can be used for explanations."
            )
        else:
            justification_parts.append(
                f"Interpretability: Lower interpretability score "
                f"({best_entry.interpretability_score:.2f}). "
                f"Consider using SHAP values for model explanations."
            )
        
        # Weight consideration
        if self.interpretability_weight > 0.4:
            justification_parts.append(
                f"Selection criteria: Interpretability was weighted at "
                f"{self.interpretability_weight:.0%}, indicating a preference "
                f"for explainable models."
            )
        else:
            justification_parts.append(
                f"Selection criteria: Performance was weighted at "
                f"{self.performance_weight:.0%}, prioritizing predictive accuracy."
            )
        
        return " ".join(justification_parts)
    
    def print_comparison(
        self,
        results: ModelComparisonResults,
        include_cv: bool = True
    ) -> None:
        """
        Print a formatted side-by-side comparison of models.
        
        Parameters:
        -----------
        results : ModelComparisonResults
            Comparison results to print
        include_cv : bool, default True
            Whether to include cross-validation metrics in the output
        """
        print("\n" + "=" * 100)
        print("MODEL COMPARISON - SIDE-BY-SIDE")
        print("=" * 100)
        
        df = results.comparison_df.copy()
        
        # Select columns to display
        display_cols = ['model_name', 'model_type', 'interpretability_score']
        
        # Add test metrics
        test_cols = [col for col in df.columns if col.startswith('test_')]
        display_cols.extend(sorted(test_cols))
        
        # Add CV metrics if requested and available
        if include_cv:
            cv_mean_cols = [col for col in df.columns if col.startswith('cv_') and col.endswith('_mean')]
            display_cols.extend(sorted(cv_mean_cols))
        
        display_cols.extend(['performance_score', 'overall_score'])
        
        # Filter to available columns
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Format DataFrame for display
        display_df = df[display_cols].copy()
        
        # Round numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(4)
        
        print("\n📊 Performance Metrics:")
        print("-" * 100)
        print(display_df.to_string(index=False))
        
        print("\n" + "=" * 100)
        print("BEST MODEL SELECTION")
        print("=" * 100)
        print(f"\n🏆 Best Model: {results.best_model_name}")
        print(f"\n📝 Justification:")
        print("-" * 100)
        # Wrap justification text
        words = results.best_model_justification.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > 90:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            print(f"  {line}")
        
        print("\n📊 Model Ranking:")
        print("-" * 100)
        for rank, model_name in enumerate(results.ranking, 1):
            score = df[df['model_name'] == model_name]['overall_score'].iloc[0]
            print(f"  {rank}. {model_name} (Score: {score:.4f})")
        
        print("=" * 100)
    
    def get_comparison_summary(
        self,
        results: ModelComparisonResults
    ) -> Dict[str, Any]:
        """
        Get a summary dictionary of the comparison results.
        
        Parameters:
        -----------
        results : ModelComparisonResults
            Comparison results
        
        Returns:
        --------
        Dict[str, Any]
            Summary dictionary
        """
        best_row = results.comparison_df[
            results.comparison_df['model_name'] == results.best_model_name
        ].iloc[0]
        
        summary = {
            'best_model': results.best_model_name,
            'best_model_type': best_row['model_type'],
            'best_overall_score': float(best_row['overall_score']),
            'best_performance_score': float(best_row['performance_score']),
            'best_interpretability_score': float(best_row['interpretability_score']),
            'primary_metric': self.primary_metric,
            'primary_metric_value': float(best_row.get(f'test_{self.primary_metric}', 0.0)),
            'justification': results.best_model_justification,
            'ranking': results.ranking,
            'total_models_compared': len(results.comparison_df)
        }
        
        return summary

