"""
Baseline Model Module

This module provides a professional, reusable class-based approach for training
and evaluating baseline Logistic Regression models for fraud detection.

The baseline model serves as an interpretable benchmark for comparing against
more complex ensemble models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import joblib
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineModelResults:
    """
    Data class to hold baseline model evaluation results.
    
    Attributes:
    -----------
    model : LogisticRegression
        Trained baseline model
    train_metrics : Dict[str, float]
        Training set metrics
    test_metrics : Dict[str, float]
        Test set metrics
    confusion_matrix : np.ndarray
        Confusion matrix on test set
    classification_report : str
        Detailed classification report
    """
    model: LogisticRegression
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str


class BaselineModel:
    """
    Professional baseline model class for fraud detection.
    
    This class provides a reusable, object-oriented approach to:
    - Train Logistic Regression as an interpretable baseline
    - Evaluate using AUC-PR, F1-Score, and Confusion Matrix
    - Handle imbalanced data with class weights
    - Provide comprehensive evaluation metrics
    
    Attributes:
    -----------
    class_weight : str or dict
        Class weights for handling imbalanced data
    random_state : int
        Random seed for reproducibility
    max_iter : int
        Maximum iterations for Logistic Regression
    model : LogisticRegression, optional
        Trained model (set after training)
    
    Example:
    --------
    >>> from src.baseline_model import BaselineModel
    >>> from src.data_preparation import DataPreparation
    >>> 
    >>> # Prepare data
    >>> prep = DataPreparation(dataset_type='ecommerce')
    >>> result = prep.prepare_and_split(df)
    >>> 
    >>> # Train baseline model
    >>> baseline = BaselineModel(class_weight='balanced', random_state=42)
    >>> baseline_results = baseline.train_and_evaluate(
    ...     result.X_train, result.y_train,
    ...     result.X_test, result.y_test
    ... )
    >>> 
    >>> # Access results
    >>> print(f"F1-Score: {baseline_results.test_metrics['f1']:.4f}")
    >>> print(f"AUC-PR: {baseline_results.test_metrics['pr_auc']:.4f}")
    """
    
    def __init__(
        self,
        class_weight: Union[str, Dict[int, float]] = 'balanced',
        random_state: int = 42,
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize BaselineModel instance.
        
        Parameters:
        -----------
        class_weight : str or dict, default 'balanced'
            Class weights for handling imbalanced data.
            Options: 'balanced', None, or custom dict {0: weight0, 1: weight1}
        random_state : int, default 42
            Random seed for reproducibility
        max_iter : int, default 1000
            Maximum iterations for Logistic Regression convergence
        solver : str, default 'lbfgs'
            Algorithm to use for optimization
            Options: 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'
        n_jobs : int, default -1
            Number of CPU cores to use (-1 for all cores)
        **kwargs
            Additional parameters to pass to LogisticRegression
        
        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        # Validate class_weight
        if class_weight not in ['balanced', None] and not isinstance(class_weight, dict):
            raise ValueError(
                f"Invalid class_weight: {class_weight}. "
                "Must be 'balanced', None, or a dict."
            )
        
        # Validate solver
        valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        if solver not in valid_solvers:
            raise ValueError(
                f"Invalid solver: {solver}. Must be one of: {valid_solvers}"
            )
        
        # Set attributes
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model: Optional[LogisticRegression] = None
        
        logger.info("Initialized BaselineModel (Logistic Regression)")
        logger.info(f"  Class weight: {class_weight}")
        logger.info(f"  Solver: {solver}, Max iterations: {max_iter}")
        logger.info(f"  Random state: {random_state}")
    
    def _create_model(self) -> LogisticRegression:
        """
        Create and configure Logistic Regression model.
        
        Returns:
        --------
        LogisticRegression
            Configured but untrained model
        """
        model = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver=self.solver,
            n_jobs=self.n_jobs,
            **self.kwargs
        )
        return model
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> LogisticRegression:
        """
        Train the baseline Logistic Regression model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target variable
        
        Returns:
        --------
        LogisticRegression
            Trained model
        
        Raises:
        -------
        ValueError
            If inputs are invalid
        RuntimeError
            If training fails
        """
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("X_train and y_train cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train must have the same length. "
                f"Got {len(X_train)} and {len(y_train)}"
            )
        
        logger.info(f"Training baseline model on {len(X_train)} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Class distribution: {np.bincount(y_train)}")
        
        # Create and train model
        try:
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
            logger.info("✓ Baseline model trained successfully")
        except Exception as e:
            error_msg = f"Error training baseline model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return self.model
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on given data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True target values
        metrics : List[str], optional
            List of metrics to compute. If None, computes all available metrics.
            Available: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of metric names and values
        
        Raises:
        -------
        ValueError
            If model is not trained or inputs are invalid
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Validate inputs
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
        
        logger.info(f"Evaluating model on {len(X)} samples")
        
        # Make predictions
        try:
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Default metrics
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        results = {}
        
        # Compute metrics
        try:
            if 'accuracy' in metrics:
                results['accuracy'] = float(accuracy_score(y, y_pred))
            
            if 'precision' in metrics:
                results['precision'] = float(precision_score(y, y_pred, zero_division=0))
            
            if 'recall' in metrics:
                results['recall'] = float(recall_score(y, y_pred, zero_division=0))
            
            if 'f1' in metrics:
                results['f1'] = float(f1_score(y, y_pred, zero_division=0))
            
            if 'roc_auc' in metrics:
                try:
                    results['roc_auc'] = float(roc_auc_score(y, y_pred_proba))
                except ValueError as e:
                    logger.warning(f"Could not compute ROC AUC: {str(e)}")
                    results['roc_auc'] = None
            
            if 'pr_auc' in metrics:
                try:
                    results['pr_auc'] = float(average_precision_score(y, y_pred_proba))
                except ValueError as e:
                    logger.warning(f"Could not compute PR AUC: {str(e)}")
                    results['pr_auc'] = None
            
            logger.info("Model evaluation completed")
            for metric, value in results.items():
                if value is not None:
                    logger.info(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            error_msg = f"Error computing metrics: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return results
    
    def get_confusion_matrix(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Get confusion matrix for the model predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True target values
        
        Returns:
        --------
        np.ndarray
            Confusion matrix (2x2 for binary classification)
            Format: [[TN, FP], [FN, TP]]
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        logger.info(f"Confusion matrix:\n{cm}")
        return cm
    
    def get_classification_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> str:
        """
        Get detailed classification report.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True target values
        
        Returns:
        --------
        str
            Classification report as formatted string
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        
        return report
    
    def train_and_evaluate(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> BaselineModelResults:
        """
        Complete pipeline: train model and evaluate on both train and test sets.
        
        This is a convenience method that combines train() and evaluate() for
        a complete workflow.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target variable
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test target variable
        metrics : List[str], optional
            List of metrics to compute
        
        Returns:
        --------
        BaselineModelResults
            Dataclass containing trained model and all evaluation results
        
        Example:
        --------
        >>> baseline = BaselineModel()
        >>> results = baseline.train_and_evaluate(X_train, y_train, X_test, y_test)
        >>> print(f"Test F1-Score: {results.test_metrics['f1']:.4f}")
        >>> print(f"Test AUC-PR: {results.test_metrics['pr_auc']:.4f}")
        """
        logger.info("Starting baseline model training and evaluation pipeline")
        
        # Step 1: Train model
        self.train(X_train, y_train)
        
        # Step 2: Evaluate on training set
        logger.info("Evaluating on training set...")
        train_metrics = self.evaluate(X_train, y_train, metrics=metrics)
        
        # Step 3: Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(X_test, y_test, metrics=metrics)
        
        # Step 4: Get confusion matrix
        cm = self.get_confusion_matrix(X_test, y_test)
        
        # Step 5: Get classification report
        report = self.get_classification_report(X_test, y_test)
        
        # Create results object
        results = BaselineModelResults(
            model=self.model,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            confusion_matrix=cm,
            classification_report=report
        )
        
        logger.info("Baseline model training and evaluation completed successfully")
        
        return results
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        file_path : str or Path
            Path where to save the model
        
        Raises:
        -------
        ValueError
            If model is not trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving baseline model to: {file_path}")
        try:
            joblib.dump(self.model, file_path)
            logger.info("✓ Model saved successfully")
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def load_model(self, file_path: Union[str, Path]) -> LogisticRegression:
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the saved model file
        
        Returns:
        --------
        LogisticRegression
            Loaded model
        
        Raises:
        -------
        FileNotFoundError
            If model file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        logger.info(f"Loading baseline model from: {file_path}")
        try:
            self.model = joblib.load(file_path)
            logger.info("✓ Model loaded successfully")
            return self.model
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (coefficients) from the trained model.
        
        Returns:
        --------
        pd.DataFrame, optional
            DataFrame with feature names and coefficients, sorted by absolute value.
            Returns None if model is not trained or features are not available.
        """
        if self.model is None:
            logger.warning("Model has not been trained. Cannot get feature importance.")
            return None
        
        try:
            coefficients = self.model.coef_[0]
            
            # Try to get feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f'Feature_{i}' for i in range(len(coefficients))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            logger.info("Feature importance extracted")
            return importance_df
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None

