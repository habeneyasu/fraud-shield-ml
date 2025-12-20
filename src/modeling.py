"""
Modeling Module

This module provides reusable functions for model training, evaluation,
and prediction with proper error handling and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    model_type: str = 'random_forest',
    model_params: Optional[Dict[str, Any]] = None,
    class_weight: Optional[Union[str, Dict[int, float]]] = 'balanced'
) -> Any:
    """
    Train a machine learning model with validation and error handling.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target variable
    model_type : str, default 'random_forest'
        Type of model: 'random_forest', 'gradient_boosting', 'logistic_regression',
                      'xgboost', or 'lightgbm'
    model_params : dict, optional
        Model-specific hyperparameters
    class_weight : str or dict, default 'balanced'
        Class weights for handling imbalanced data
    
    Returns:
    --------
    Trained model object
    
    Raises:
    -------
    ValueError
        If inputs are invalid or model_type is unsupported
    """
    try:
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("X_train and y_train cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train must have the same length. Got {len(X_train)} and {len(y_train)}")
        
        if model_params is None:
            model_params = {}
        
        logger.info(f"Training {model_type} model with {len(X_train)} samples")
        
        # Initialize and train model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
                **model_params
            )
        
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                random_state=42,
                **model_params
            )
        
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                class_weight=class_weight,
                random_state=42,
                max_iter=1000,
                n_jobs=-1,
                **model_params
            )
        
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                **model_params
            )
        
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
                **model_params
            )
        
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                "Must be one of: 'random_forest', 'gradient_boosting', "
                "'logistic_regression', 'xgboost', 'lightgbm'"
            )
        
        # Train the model
        try:
            model.fit(X_train, y_train)
            logger.info(f"Successfully trained {model_type} model")
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return model
        
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error training model: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def evaluate_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate a trained model with validation and error handling.
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model to evaluate
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        True target values
    metrics : list, optional
        List of metrics to compute. If None, computes all available metrics
    
    Returns:
    --------
    dict
        Dictionary of metric names and values
    
    Raises:
    -------
    ValueError
        If inputs are invalid
    """
    try:
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
            raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
        
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
        
        logger.info(f"Evaluating model on {len(X)} samples")
        
        # Make predictions
        try:
            y_pred = model.predict(X)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Default metrics
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        results = {}
        
        # Compute metrics
        try:
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(y, y_pred)
            
            if 'precision' in metrics:
                results['precision'] = precision_score(y, y_pred, zero_division=0)
            
            if 'recall' in metrics:
                results['recall'] = recall_score(y, y_pred, zero_division=0)
            
            if 'f1' in metrics:
                results['f1'] = f1_score(y, y_pred, zero_division=0)
            
            if 'roc_auc' in metrics and y_pred_proba is not None:
                try:
                    results['roc_auc'] = roc_auc_score(y, y_pred_proba)
                except ValueError as e:
                    logger.warning(f"Could not compute ROC AUC: {str(e)}")
                    results['roc_auc'] = None
            
            logger.info("Model evaluation completed successfully")
            for metric, value in results.items():
                if value is not None:
                    logger.info(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            error_msg = f"Error computing metrics: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return results
        
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error evaluating model: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_classification_report(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> str:
    """
    Generate a detailed classification report with error handling.
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        True target values
    
    Returns:
    --------
    str
        Classification report as string
    """
    try:
        if model is None or X is None or y is None:
            raise ValueError("Model, X, and y cannot be None")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        
        logger.info("Generated classification report")
        return report
        
    except Exception as e:
        error_msg = f"Error generating classification report: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_confusion_matrix(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Get confusion matrix with error handling.
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        True target values
    
    Returns:
    --------
    np.ndarray
        Confusion matrix
    """
    try:
        if model is None or X is None or y is None:
            raise ValueError("Model, X, and y cannot be None")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        logger.info("Generated confusion matrix")
        return cm
        
    except Exception as e:
        error_msg = f"Error generating confusion matrix: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def save_model(
    model: Any,
    file_path: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Save a trained model to disk with error handling.
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model to save
    file_path : str or Path
        Path where to save the model
    create_dirs : bool, default True
        Whether to create parent directories if they don't exist
    
    Raises:
    -------
    ValueError
        If model is None or invalid
    """
    try:
        if model is None:
            raise ValueError("Model cannot be None")
        
        file_path = Path(file_path)
        
        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {file_path}")
        
        try:
            joblib.dump(model, file_path)
        except Exception as e:
            error_msg = f"Error saving model to {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Verify file was created
        if not file_path.exists():
            error_msg = f"Model file was not created: {file_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"Successfully saved model to: {file_path}")
        
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error saving model: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_model(
    file_path: Union[str, Path]
) -> Any:
    """
    Load a saved model from disk with error handling.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the saved model file
    
    Returns:
    --------
    Loaded model object
    
    Raises:
    -------
    FileNotFoundError
        If model file does not exist
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"Model file not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading model from: {file_path}")
        
        try:
            model = joblib.load(file_path)
        except Exception as e:
            error_msg = f"Error loading model from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        logger.info("Successfully loaded model")
        return model
        
    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error loading model: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def cross_validate_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> Dict[str, float]:
    """
    Perform cross-validation on a model with error handling.
    
    Parameters:
    -----------
    model : sklearn model object
        Model to cross-validate
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    cv : int, default 5
        Number of cross-validation folds
    scoring : str, default 'roc_auc'
        Scoring metric
    
    Returns:
    --------
    dict
        Dictionary with mean and std of cross-validation scores
    """
    try:
        if model is None or X is None or y is None:
            raise ValueError("Model, X, and y cannot be None")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
        
        if cv < 2:
            raise ValueError(f"cv must be at least 2. Got {cv}")
        
        logger.info(f"Performing {cv}-fold cross-validation")
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        except Exception as e:
            error_msg = f"Error during cross-validation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        results = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"Cross-validation {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
        
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error in cross-validation: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def predict(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions using a trained model with error handling.
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model
    X : pd.DataFrame or np.ndarray
        Features to predict on
    return_proba : bool, default False
        Whether to return prediction probabilities
    
    Returns:
    --------
    np.ndarray or tuple
        Predictions, and optionally probabilities
    """
    try:
        if model is None:
            raise ValueError("Model cannot be None")
        
        if X is None:
            raise ValueError("X cannot be None")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        if X.size == 0:
            raise ValueError("X cannot be empty")
        
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
        
        logger.info(f"Making predictions on {len(X)} samples")
        
        try:
            y_pred = model.predict(X)
            
            if return_proba:
                if not hasattr(model, 'predict_proba'):
                    raise ValueError("Model does not support probability predictions")
                y_proba = model.predict_proba(X)
                logger.info("Predictions and probabilities generated")
                return y_pred, y_proba
            else:
                logger.info("Predictions generated")
                return y_pred
        
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error making predictions: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

