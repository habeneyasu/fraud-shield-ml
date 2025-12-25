"""
Ensemble Model Module

This module provides a professional, reusable class-based approach for training
and evaluating ensemble models (Random Forest, XGBoost, LightGBM) with
hyperparameter tuning for fraud detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    make_scorer
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleModelResults:
    """
    Data class to hold ensemble model evaluation results.
    
    Attributes:
    -----------
    model : Any
        Trained ensemble model
    best_params : Dict[str, Any]
        Best hyperparameters found during tuning
    train_metrics : Dict[str, float]
        Training set metrics
    test_metrics : Dict[str, float]
        Test set metrics
    confusion_matrix : np.ndarray
        Confusion matrix on test set
    classification_report : str
        Detailed classification report
    cv_results : Optional[Dict[str, Any]]
        Cross-validation results from hyperparameter tuning
    """
    model: Any
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    cv_results: Optional[Dict[str, Any]] = None


class EnsembleModel:
    """
    Professional ensemble model class for fraud detection.
    
    This class provides a reusable, object-oriented approach to:
    - Train ensemble models (Random Forest, XGBoost, LightGBM)
    - Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
    - Evaluate using AUC-PR, F1-Score, and Confusion Matrix
    - Handle imbalanced data with class weights
    - Provide comprehensive evaluation metrics
    
    Attributes:
    -----------
    model_type : str
        Type of ensemble model: 'random_forest', 'xgboost', or 'lightgbm'
    class_weight : str or dict
        Class weights for handling imbalanced data
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of CPU cores to use
    model : Any, optional
        Trained model (set after training)
    best_params : Dict[str, Any], optional
        Best hyperparameters (set after tuning)
    
    Example:
    --------
    >>> from src.ensemble_model import EnsembleModel
    >>> from src.data_preparation import DataPreparation
    >>> 
    >>> # Prepare data
    >>> prep = DataPreparation(dataset_type='ecommerce')
    >>> result = prep.prepare_and_split(df)
    >>> 
    >>> # Train ensemble model with hyperparameter tuning
    >>> ensemble = EnsembleModel(
    ...     model_type='random_forest',
    ...     class_weight='balanced',
    ...     random_state=42
    ... )
    >>> 
    >>> # Define hyperparameter grid
    >>> param_grid = {
    ...     'n_estimators': [100, 200],
    ...     'max_depth': [10, 20, None]
    ... }
    >>> 
    >>> # Train and evaluate
    >>> results = ensemble.train_and_evaluate(
    ...     result.X_train, result.y_train,
    ...     result.X_test, result.y_test,
    ...     param_grid=param_grid,
    ...     cv=5
    ... )
    >>> 
    >>> # Access results
    >>> print(f"Best params: {results.best_params}")
    >>> print(f"F1-Score: {results.test_metrics['f1']:.4f}")
    >>> print(f"AUC-PR: {results.test_metrics['pr_auc']:.4f}")
    """
    
    # Supported model types
    SUPPORTED_MODELS = ['random_forest', 'xgboost', 'lightgbm']
    
    # Default hyperparameter grids for each model type
    DEFAULT_PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
    }
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        class_weight: Optional[Union[str, Dict[int, float]]] = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize EnsembleModel instance.
        
        Parameters:
        -----------
        model_type : str, default 'random_forest'
            Type of ensemble model: 'random_forest', 'xgboost', or 'lightgbm'
        class_weight : str or dict, optional, default 'balanced'
            Class weights for handling imbalanced data.
            Options: 'balanced', None, or custom dict {0: weight0, 1: weight1}
            Note: XGBoost and LightGBM use scale_pos_weight instead
        random_state : int, default 42
            Random seed for reproducibility
        n_jobs : int, default -1
            Number of CPU cores to use (-1 for all cores)
        **kwargs
            Additional parameters to pass to the model
        
        Raises:
        -------
        ValueError
            If model_type is invalid or parameters are invalid
        """
        # Validate model_type
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Must be one of: {self.SUPPORTED_MODELS}"
            )
        
        # Validate class_weight
        if class_weight not in ['balanced', None] and not isinstance(class_weight, dict):
            raise ValueError(
                f"Invalid class_weight: {class_weight}. "
                "Must be 'balanced', None, or a dict."
            )
        
        # Set attributes
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model: Optional[Any] = None
        self.best_params: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized EnsembleModel ({model_type})")
        logger.info(f"  Class weight: {class_weight}")
        logger.info(f"  Random state: {random_state}, n_jobs: {n_jobs}")
    
    def _create_model(self, **params) -> Any:
        """
        Create and configure ensemble model.
        
        Parameters:
        -----------
        **params
            Model-specific hyperparameters
        
        Returns:
        --------
        Any
            Configured but untrained model
        """
        # Merge default kwargs with provided params
        model_params = {**self.kwargs, **params}
        
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **model_params
            )
        
        elif self.model_type == 'xgboost':
            # XGBoost doesn't use class_weight, use scale_pos_weight instead
            if self.class_weight == 'balanced':
                # Calculate scale_pos_weight from class distribution
                # This will be set during training if needed
                pass
            
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=self.n_jobs,
                **model_params
            )
        
        elif self.model_type == 'lightgbm':
            # LightGBM doesn't use class_weight, use scale_pos_weight instead
            model = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                n_jobs=self.n_jobs,
                **model_params
            )
        
        return model
    
    def _calculate_scale_pos_weight(
        self,
        y: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate scale_pos_weight for XGBoost/LightGBM from class distribution.
        
        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Target variable
        
        Returns:
        --------
        float
            Scale positive weight
        """
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        class_counts = np.bincount(y)
        if len(class_counts) < 2:
            return 1.0
        return class_counts[0] / class_counts[1]
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        **model_params
    ) -> Any:
        """
        Train the ensemble model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target variable
        **model_params
            Model-specific hyperparameters
        
        Returns:
        --------
        Any
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
        
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Class distribution: {np.bincount(y_train)}")
        
        # Handle scale_pos_weight for XGBoost/LightGBM
        if self.class_weight == 'balanced' and self.model_type in ['xgboost', 'lightgbm']:
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            model_params['scale_pos_weight'] = scale_pos_weight
            logger.info(f"  Using scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Create and train model
        try:
            self.model = self._create_model(**model_params)
            self.model.fit(X_train, y_train)
            logger.info(f"✓ {self.model_type} model trained successfully")
        except Exception as e:
            error_msg = f"Error training {self.model_type} model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return self.model
    
    def tune_hyperparameters(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        scoring: str = 'f1',
        search_type: str = 'grid',
        n_iter: int = 20,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target variable
        param_grid : Dict[str, List[Any]], optional
            Hyperparameter grid to search. If None, uses default grid for model_type
        cv : int, default 5
            Number of cross-validation folds
        scoring : str, default 'f1'
            Scoring metric for hyperparameter selection
        search_type : str, default 'grid'
            Type of search: 'grid' (GridSearchCV) or 'random' (RandomizedSearchCV)
        n_iter : int, default 20
            Number of iterations for RandomizedSearchCV (ignored for GridSearchCV)
        verbose : int, default 1
            Verbosity level
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing best parameters and CV results
        
        Raises:
        -------
        ValueError
            If inputs are invalid
        """
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("X_train and y_train cannot be empty")
        
        if search_type not in ['grid', 'random']:
            raise ValueError(f"Invalid search_type: {search_type}. Must be 'grid' or 'random'")
        
        # Use default param grid if not provided
        if param_grid is None:
            param_grid = self.DEFAULT_PARAM_GRIDS[self.model_type].copy()
            logger.info(f"Using default parameter grid for {self.model_type}")
        
        # Handle scale_pos_weight for XGBoost/LightGBM
        if self.class_weight == 'balanced' and self.model_type in ['xgboost', 'lightgbm']:
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            if 'scale_pos_weight' not in param_grid:
                param_grid['scale_pos_weight'] = [scale_pos_weight]
            logger.info(f"  Using scale_pos_weight: {scale_pos_weight:.4f}")
        
        logger.info(f"Starting hyperparameter tuning ({search_type} search)")
        logger.info(f"  Parameter grid: {list(param_grid.keys())}")
        logger.info(f"  CV folds: {cv}, Scoring: {scoring}")
        
        # Create base model
        base_model = self._create_model()
        
        # Create cross-validation splitter
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Create scorer
        scorer = make_scorer(f1_score, zero_division=0) if scoring == 'f1' else scoring
        
        # Perform search
        try:
            if search_type == 'grid':
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_splitter,
                    scoring=scorer,
                    n_jobs=self.n_jobs,
                    verbose=verbose,
                    return_train_score=True
                )
            else:  # random
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv_splitter,
                    scoring=scorer,
                    n_jobs=self.n_jobs,
                    verbose=verbose,
                    random_state=self.random_state,
                    return_train_score=True
                )
            
            search.fit(X_train, y_train)
            
            # Store best model and parameters
            self.model = search.best_estimator_
            self.best_params = search.best_params_
            
            logger.info(f"✓ Hyperparameter tuning completed")
            logger.info(f"  Best score ({scoring}): {search.best_score_:.4f}")
            logger.info(f"  Best parameters: {self.best_params}")
            
            return {
                'best_params': self.best_params,
                'best_score': search.best_score_,
                'cv_results': {
                    'mean_test_score': search.cv_results_['mean_test_score'],
                    'std_test_score': search.cv_results_['std_test_score'],
                    'params': search.cv_results_['params']
                }
            }
        
        except Exception as e:
            error_msg = f"Error during hyperparameter tuning: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
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
            raise ValueError("Model has not been trained. Call train() or tune_hyperparameters() first.")
        
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
        """Get confusion matrix for the model predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        
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
        """Get detailed classification report."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        
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
        param_grid: Optional[Dict[str, List[Any]]] = None,
        tune_hyperparameters: bool = True,
        cv: int = 5,
        scoring: str = 'f1',
        search_type: str = 'grid',
        n_iter: int = 20,
        metrics: Optional[List[str]] = None
    ) -> EnsembleModelResults:
        """
        Complete pipeline: tune hyperparameters, train model, and evaluate.
        
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
        param_grid : Dict[str, List[Any]], optional
            Hyperparameter grid. If None, uses default for model_type
        tune_hyperparameters : bool, default True
            Whether to perform hyperparameter tuning
        cv : int, default 5
            Number of CV folds for hyperparameter tuning
        scoring : str, default 'f1'
            Scoring metric for hyperparameter selection
        search_type : str, default 'grid'
            Type of search: 'grid' or 'random'
        n_iter : int, default 20
            Number of iterations for RandomizedSearchCV
        metrics : List[str], optional
            List of metrics to compute
        
        Returns:
        --------
        EnsembleModelResults
            Dataclass containing trained model and all evaluation results
        """
        logger.info("Starting ensemble model training and evaluation pipeline")
        
        cv_results = None
        
        # Step 1: Hyperparameter tuning (if requested)
        if tune_hyperparameters:
            logger.info("Step 1: Hyperparameter tuning...")
            tuning_results = self.tune_hyperparameters(
                X_train, y_train,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                search_type=search_type,
                n_iter=n_iter
            )
            cv_results = tuning_results.get('cv_results')
        else:
            # Train without tuning
            logger.info("Step 1: Training model (no hyperparameter tuning)...")
            if param_grid:
                # Use first value from each param in grid as defaults
                default_params = {k: v[0] for k, v in param_grid.items()}
                self.train(X_train, y_train, **default_params)
            else:
                self.train(X_train, y_train)
            self.best_params = {}
        
        # Step 2: Evaluate on training set
        logger.info("Step 2: Evaluating on training set...")
        train_metrics = self.evaluate(X_train, y_train, metrics=metrics)
        
        # Step 3: Evaluate on test set
        logger.info("Step 3: Evaluating on test set...")
        test_metrics = self.evaluate(X_test, y_test, metrics=metrics)
        
        # Step 4: Get confusion matrix
        cm = self.get_confusion_matrix(X_test, y_test)
        
        # Step 5: Get classification report
        report = self.get_classification_report(X_test, y_test)
        
        # Create results object
        results = EnsembleModelResults(
            model=self.model,
            best_params=self.best_params or {},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            confusion_matrix=cm,
            classification_report=report,
            cv_results=cv_results
        )
        
        logger.info("Ensemble model training and evaluation completed successfully")
        
        return results
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving ensemble model to: {file_path}")
        try:
            joblib.dump(self.model, file_path)
            logger.info("✓ Model saved successfully")
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def load_model(self, file_path: Union[str, Path]) -> Any:
        """Load a trained model from disk."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        logger.info(f"Loading ensemble model from: {file_path}")
        try:
            self.model = joblib.load(file_path)
            logger.info("✓ Model loaded successfully")
            return self.model
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the trained model."""
        if self.model is None:
            logger.warning("Model has not been trained.")
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0])
            else:
                logger.warning("Model does not support feature importance.")
                return None
            
            # Try to get feature names
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("Feature importance extracted")
            return importance_df
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None

