"""
Cross-Validation Module

This module provides a professional, reusable class-based approach for performing
Stratified K-Fold cross-validation with comprehensive metric reporting for fraud
detection models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, make_scorer
)
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResults:
    """
    Data class to hold cross-validation results.
    
    Attributes:
    -----------
    metrics_summary : Dict[str, Dict[str, float]]
        Summary statistics for each metric with 'mean' and 'std'
        Format: {'metric_name': {'mean': float, 'std': float}}
    fold_results : List[Dict[str, float]]
        Detailed results for each fold
        Format: [{'fold': int, 'metric1': float, 'metric2': float, ...}, ...]
    n_folds : int
        Number of folds used
    """
    metrics_summary: Dict[str, Dict[str, float]]
    fold_results: List[Dict[str, float]]
    n_folds: int


class CrossValidator:
    """
    Professional cross-validation class for fraud detection models.
    
    This class provides a reusable, object-oriented approach to:
    - Perform Stratified K-Fold cross-validation (k=5 by default)
    - Compute multiple metrics (AUC-PR, F1-Score, etc.) across folds
    - Report mean and standard deviation for each metric
    - Work with both model classes (BaselineModel, EnsembleModel) and raw sklearn models
    
    Attributes:
    -----------
    n_folds : int
        Number of folds for cross-validation (default: 5)
    random_state : int
        Random seed for reproducibility
    metrics : List[str]
        List of metrics to compute
    n_jobs : int
        Number of CPU cores to use
    
    Example:
    --------
    >>> from src.cross_validation import CrossValidator
    >>> from src.baseline_model import BaselineModel
    >>> 
    >>> # Create model instance
    >>> baseline = BaselineModel(class_weight='balanced', random_state=42)
    >>> 
    >>> # Perform cross-validation
    >>> cv = CrossValidator(n_folds=5, random_state=42)
    >>> results = cv.cross_validate(
    ...     baseline, X_train, y_train,
    ...     metrics=['f1', 'pr_auc', 'roc_auc', 'precision', 'recall']
    ... )
    >>> 
    >>> # Access results
    >>> print(f"F1-Score: {results.metrics_summary['f1']['mean']:.4f} "
    ...       f"(+/- {results.metrics_summary['f1']['std']:.4f})")
    >>> print(f"AUC-PR: {results.metrics_summary['pr_auc']['mean']:.4f} "
    ...       f"(+/- {results.metrics_summary['pr_auc']['std']:.4f})")
    """
    
    # Available metrics
    AVAILABLE_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        metrics: Optional[List[str]] = None,
        n_jobs: int = -1
    ):
        """
        Initialize CrossValidator instance.
        
        Parameters:
        -----------
        n_folds : int, default 5
            Number of folds for Stratified K-Fold cross-validation
        random_state : int, default 42
            Random seed for reproducibility
        metrics : List[str], optional
            List of metrics to compute. If None, computes all available metrics.
            Available: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'
        n_jobs : int, default -1
            Number of CPU cores to use
        
        Raises:
        -------
        ValueError
            If n_folds < 2 or metrics are invalid
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2. Got {n_folds}")
        
        if metrics is None:
            metrics = self.AVAILABLE_METRICS.copy()
        else:
            invalid_metrics = set(metrics) - set(self.AVAILABLE_METRICS)
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics: {invalid_metrics}. "
                    f"Available: {self.AVAILABLE_METRICS}"
                )
        
        self.n_folds = n_folds
        self.random_state = random_state
        self.metrics = metrics
        self.n_jobs = n_jobs
        
        logger.info(f"Initialized CrossValidator")
        logger.info(f"  n_folds: {n_folds}, random_state: {random_state}")
        logger.info(f"  metrics: {metrics}")
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all requested metrics for given predictions.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted target values
        y_pred_proba : np.ndarray, optional
            Predicted probabilities (for ROC-AUC and PR-AUC)
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of metric names and values
        """
        results = {}
        
        try:
            if 'accuracy' in self.metrics:
                results['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            if 'precision' in self.metrics:
                results['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            
            if 'recall' in self.metrics:
                results['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            
            if 'f1' in self.metrics:
                results['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
            
            if 'roc_auc' in self.metrics and y_pred_proba is not None:
                try:
                    results['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
                except ValueError:
                    logger.warning("Could not compute ROC AUC (possibly only one class present)")
                    results['roc_auc'] = None
            
            if 'pr_auc' in self.metrics and y_pred_proba is not None:
                try:
                    results['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
                except ValueError:
                    logger.warning("Could not compute PR AUC (possibly only one class present)")
                    results['pr_auc'] = None
        
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise
        
        return results
    
    def _get_model_from_instance(
        self,
        model_instance: Any
    ) -> Tuple[Any, bool]:
        """
        Extract sklearn model from model class instance or return model as-is.
        
        Parameters:
        -----------
        model_instance : Any
            Either a model class instance (BaselineModel, EnsembleModel) or raw sklearn model
        
        Returns:
        --------
        Tuple[Any, bool]
            (model, is_class_instance) - model object and flag indicating if it's a class instance
        """
        # Check if it's a BaselineModel or EnsembleModel instance
        if hasattr(model_instance, 'model') and model_instance.model is not None:
            # It's a model class instance with a trained model
            return model_instance.model, True
        elif hasattr(model_instance, 'train'):
            # It's a model class instance but not yet trained
            return model_instance, True
        else:
            # Assume it's a raw sklearn model
            return model_instance, False
    
    def _train_and_predict_fold(
        self,
        model: Any,
        X_train_fold: np.ndarray,
        y_train_fold: np.ndarray,
        X_val_fold: np.ndarray,
        is_class_instance: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Train model on fold training data and predict on fold validation data.
        
        Parameters:
        -----------
        model : Any
            Model instance (class or sklearn model)
        X_train_fold : np.ndarray
            Training features for this fold
        y_train_fold : np.ndarray
            Training target for this fold
        X_val_fold : np.ndarray
            Validation features for this fold
        is_class_instance : bool
            Whether model is a class instance (BaselineModel/EnsembleModel)
        
        Returns:
        --------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (y_pred, y_pred_proba) predictions
        """
        if is_class_instance:
            # It's a model class instance - train and predict
            model.train(X_train_fold, y_train_fold)
            y_pred = model.model.predict(X_val_fold)
            y_pred_proba = None
            if hasattr(model.model, 'predict_proba'):
                y_pred_proba = model.model.predict_proba(X_val_fold)[:, 1]
        else:
            # It's a raw sklearn model - clone, train, and predict
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            y_pred = fold_model.predict(X_val_fold)
            y_pred_proba = None
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_val_fold)[:, 1]
        
        return y_pred, y_pred_proba
    
    def cross_validate(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> CrossValidationResults:
        """
        Perform Stratified K-Fold cross-validation.
        
        Parameters:
        -----------
        model : Any
            Model to cross-validate. Can be:
            - Model class instance (BaselineModel, EnsembleModel) - will train on each fold
            - Raw sklearn model - will be cloned and trained on each fold
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            Target variable
        metrics : List[str], optional
            Override default metrics for this call
        
        Returns:
        --------
        CrossValidationResults
            Dataclass containing cross-validation results with mean and std for each metric
        
        Raises:
        -------
        ValueError
            If inputs are invalid
        """
        # Validate inputs
        if model is None:
            raise ValueError("Model cannot be None")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got {len(X)} and {len(y)}"
            )
        
        # Override metrics if provided
        if metrics is not None:
            invalid_metrics = set(metrics) - set(self.AVAILABLE_METRICS)
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics: {invalid_metrics}. "
                    f"Available: {self.AVAILABLE_METRICS}"
                )
            original_metrics = self.metrics
            self.metrics = metrics
        else:
            original_metrics = None
        
        logger.info(f"Starting {self.n_folds}-fold Stratified K-Fold cross-validation")
        logger.info(f"  Data shape: {X.shape}")
        logger.info(f"  Metrics: {self.metrics}")
        
        # Get model (extract from class instance if needed)
        model_obj, is_class_instance = self._get_model_from_instance(model)
        
        # Create Stratified K-Fold splitter
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Store results for each fold
        fold_results = []
        
        # Perform cross-validation
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                logger.info(f"Processing fold {fold_idx}/{self.n_folds}...")
                
                # Split data for this fold
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                logger.debug(
                    f"  Fold {fold_idx}: Train={len(X_train_fold)}, "
                    f"Val={len(X_val_fold)}"
                )
                
                # Train and predict
                y_pred, y_pred_proba = self._train_and_predict_fold(
                    model_obj if not is_class_instance else model,
                    X_train_fold, y_train_fold,
                    X_val_fold, is_class_instance
                )
                
                # Compute metrics for this fold
                fold_metrics = self._compute_metrics(y_val_fold, y_pred, y_pred_proba)
                fold_metrics['fold'] = fold_idx
                fold_results.append(fold_metrics)
                
                logger.info(
                    f"  Fold {fold_idx} - F1: {fold_metrics.get('f1', 'N/A'):.4f}, "
                    f"PR-AUC: {fold_metrics.get('pr_auc', 'N/A'):.4f if fold_metrics.get('pr_auc') is not None else 'N/A'}"
                )
        
        except Exception as e:
            error_msg = f"Error during cross-validation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        finally:
            # Restore original metrics if overridden
            if original_metrics is not None:
                self.metrics = original_metrics
        
        # Compute summary statistics (mean and std) for each metric
        metrics_summary = {}
        for metric in self.metrics:
            metric_values = [
                fold[metric] for fold in fold_results
                if fold.get(metric) is not None
            ]
            
            if metric_values:
                metrics_summary[metric] = {
                    'mean': float(np.mean(metric_values)),
                    'std': float(np.std(metric_values)),
                    'min': float(np.min(metric_values)),
                    'max': float(np.max(metric_values))
                }
            else:
                metrics_summary[metric] = {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None
                }
        
        # Log summary
        logger.info("Cross-validation completed")
        logger.info("Summary statistics:")
        for metric, stats in metrics_summary.items():
            if stats['mean'] is not None:
                logger.info(
                    f"  {metric.upper()}: {stats['mean']:.4f} "
                    f"(+/- {stats['std']:.4f}) "
                    f"[min: {stats['min']:.4f}, max: {stats['max']:.4f}]"
                )
        
        # Create results object
        results = CrossValidationResults(
            metrics_summary=metrics_summary,
            fold_results=fold_results,
            n_folds=self.n_folds
        )
        
        return results
    
    def print_summary(self, results: CrossValidationResults) -> None:
        """
        Print a formatted summary of cross-validation results.
        
        Parameters:
        -----------
        results : CrossValidationResults
            Cross-validation results to print
        """
        print("\n" + "=" * 80)
        print(f"Cross-Validation Results ({results.n_folds}-Fold Stratified K-Fold)")
        print("=" * 80)
        
        print("\n📊 Summary Statistics (Mean ± Std):")
        print("-" * 80)
        
        for metric, stats in results.metrics_summary.items():
            if stats['mean'] is not None:
                print(
                    f"  {metric.upper():12s}: {stats['mean']:7.4f} "
                    f"(± {stats['std']:7.4f}) "
                    f"[Range: {stats['min']:.4f} - {stats['max']:.4f}]"
                )
            else:
                print(f"  {metric.upper():12s}: N/A")
        
        print("\n📊 Per-Fold Results:")
        print("-" * 80)
        for fold_result in results.fold_results:
            fold_num = fold_result.pop('fold')
            metrics_str = ", ".join([
                f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in fold_result.items()
                if v is not None
            ])
            print(f"  Fold {fold_num}: {metrics_str}")
            fold_result['fold'] = fold_num  # Restore for potential future use
        
        print("=" * 80)

