"""
Model Explainability Module

This module provides a professional, reusable class-based approach for explaining
model predictions using SHAP (SHapley Additive exPlanations) and built-in feature
importance methods.

Designed for fraud detection models to provide actionable business insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass

# Optional SHAP import (handle gracefully if not available)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SHAP not available. Install with: pip install shap")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


@dataclass
class FeatureImportanceResults:
    """
    Data class to hold feature importance results.
    
    Attributes:
    -----------
    importance_df : pd.DataFrame
        DataFrame with features and their importance scores, sorted by importance
    top_features : List[str]
        List of top N feature names
    method : str
        Method used to extract importance ('builtin', 'shap', 'coefficients')
    """
    importance_df: pd.DataFrame
    top_features: List[str]
    method: str


@dataclass
class CaseStudyInstance:
    """
    Data class to hold information about a case study instance.
    
    Attributes:
    -----------
    index : int
        Index of the instance in the dataset
    prediction : int
        Model prediction (0 or 1)
    actual : int
        Actual label (0 or 1)
    case_type : str
        Type of case: 'TP', 'FP', 'FN', 'TN'
    probability : float
        Prediction probability for positive class
    """
    index: int
    prediction: int
    actual: int
    case_type: str
    probability: float


@dataclass
class ExplainabilityResults:
    """
    Data class to hold comprehensive explainability results.
    
    Attributes:
    -----------
    feature_importance : FeatureImportanceResults
        Feature importance results
    shap_values : Optional[np.ndarray]
        SHAP values if computed
    shap_explainer : Optional[Any]
        SHAP explainer object
    visualizations : Dict[str, Path]
        Dictionary mapping visualization names to file paths
    case_studies : Optional[List[CaseStudyInstance]]
        List of case study instances (TP, FP, FN)
    """
    feature_importance: FeatureImportanceResults
    shap_values: Optional[np.ndarray] = None
    shap_explainer: Optional[Any] = None
    visualizations: Optional[Dict[str, Path]] = None
    case_studies: Optional[List[CaseStudyInstance]] = None


class ModelExplainability:
    """
    Professional model explainability class for fraud detection models.
    
    This class provides a reusable, object-oriented approach to:
    - Extract built-in feature importance from ensemble models
    - Visualize top features with professional charts
    - Generate SHAP explanations (global and local)
    - Provide actionable business insights
    
    Attributes:
    -----------
    model : Any
        Trained model to explain
    feature_names : Optional[List[str]]
        Names of features (for better visualization)
    random_state : int
        Random seed for reproducibility
    
    Example:
    --------
    >>> from src.model_explainability import ModelExplainability
    >>> from src.ensemble_model import EnsembleModel
    >>> 
    >>> # Train model
    >>> ensemble = EnsembleModel(model_type='random_forest')
    >>> ensemble.train(X_train, y_train)
    >>> 
    >>> # Create explainer
    >>> explainer = ModelExplainability(
    ...     model=ensemble.model,
    ...     feature_names=X_train.columns.tolist()
    ... )
    >>> 
    >>> # Extract feature importance
    >>> importance = explainer.extract_feature_importance()
    >>> 
    >>> # Visualize top 10 features
    >>> explainer.visualize_feature_importance(
    ...     importance, top_n=10, save_path='visualizations/feature_importance.png'
    ... )
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize ModelExplainability instance.
        
        Parameters:
        -----------
        model : Any
            Trained model to explain (must have predict or predict_proba method)
        feature_names : List[str], optional
            Names of features. If None, will try to extract from model or use generic names
        random_state : int, default 42
            Random seed for reproducibility
        
        Raises:
        -------
        ValueError
            If model is None or invalid
        """
        if model is None:
            raise ValueError("Model cannot be None")
        
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
        
        self.model = model
        self.random_state = random_state
        
        # Try to get feature names
        if feature_names is None:
            self.feature_names = self._extract_feature_names()
        else:
            self.feature_names = feature_names
        
        logger.info("Initialized ModelExplainability")
        logger.info(f"  Number of features: {len(self.feature_names)}")
        logger.info(f"  SHAP available: {SHAP_AVAILABLE}")
    
    def _extract_feature_names(self) -> List[str]:
        """
        Extract feature names from model or use generic names.
        
        Returns:
        --------
        List[str]
            List of feature names
        """
        # Try to get feature names from model
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, 'feature_importances_'):
            n_features = len(self.model.feature_importances_)
            return [f'Feature_{i}' for i in range(n_features)]
        elif hasattr(self.model, 'coef_'):
            n_features = len(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else len(self.model.coef_)
            return [f'Feature_{i}' for i in range(n_features)]
        else:
            # Try to infer from model attributes
            logger.warning("Could not extract feature names. Using generic names.")
            return [f'Feature_{i}' for i in range(100)]  # Default, will be adjusted
    
    def extract_feature_importance(
        self,
        method: str = 'auto'
    ) -> FeatureImportanceResults:
        """
        Extract feature importance from the model.
        
        Supports multiple methods:
        - 'builtin': Uses model's built-in feature_importances_ (for tree-based models)
        - 'coefficients': Uses model coefficients (for linear models)
        - 'auto': Automatically selects best method
        
        Parameters:
        -----------
        method : str, default 'auto'
            Method to use: 'auto', 'builtin', or 'coefficients'
        
        Returns:
        --------
        FeatureImportanceResults
            Dataclass containing feature importance DataFrame and metadata
        
        Raises:
        -------
        ValueError
            If method is invalid or model doesn't support it
        """
        logger.info(f"Extracting feature importance using method: {method}")
        
        importance_scores = None
        extraction_method = None
        
        # Auto-detect method
        if method == 'auto':
            if hasattr(self.model, 'feature_importances_'):
                method = 'builtin'
            elif hasattr(self.model, 'coef_'):
                method = 'coefficients'
            else:
                raise ValueError(
                    "Model does not support feature importance extraction. "
                    "Model must have 'feature_importances_' or 'coef_' attribute."
                )
        
        # Extract importance based on method
        if method == 'builtin':
            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model does not have 'feature_importances_' attribute")
            importance_scores = self.model.feature_importances_
            extraction_method = 'builtin'
            logger.info("Using built-in feature_importances_")
        
        elif method == 'coefficients':
            if not hasattr(self.model, 'coef_'):
                raise ValueError("Model does not have 'coef_' attribute")
            # Use absolute values of coefficients
            coef = self.model.coef_
            if len(coef.shape) > 1:
                importance_scores = np.abs(coef[0])
            else:
                importance_scores = np.abs(coef)
            extraction_method = 'coefficients'
            logger.info("Using model coefficients (absolute values)")
        
        else:
            raise ValueError(f"Invalid method: {method}. Must be 'auto', 'builtin', or 'coefficients'")
        
        # Adjust feature names if needed
        if len(self.feature_names) != len(importance_scores):
            logger.warning(
                f"Feature names length ({len(self.feature_names)}) doesn't match "
                f"importance scores length ({len(importance_scores)}). Adjusting..."
            )
            self.feature_names = [f'Feature_{i}' for i in range(len(importance_scores))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance_scores)],
            'importance': importance_scores,
            'abs_importance': np.abs(importance_scores)
        }).sort_values('abs_importance', ascending=False)
        
        # Reset index
        importance_df = importance_df.reset_index(drop=True)
        
        logger.info(f"✓ Extracted feature importance for {len(importance_df)} features")
        logger.info(f"  Top feature: {importance_df.iloc[0]['feature']} "
                   f"(importance: {importance_df.iloc[0]['importance']:.4f})")
        
        results = FeatureImportanceResults(
            importance_df=importance_df,
            top_features=importance_df['feature'].tolist(),
            method=extraction_method
        )
        
        return results
    
    def visualize_feature_importance(
        self,
        importance_results: FeatureImportanceResults,
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> Path:
        """
        Visualize top N most important features.
        
        Parameters:
        -----------
        importance_results : FeatureImportanceResults
            Feature importance results from extract_feature_importance()
        top_n : int, default 10
            Number of top features to visualize
        figsize : Tuple[int, int], default (10, 6)
            Figure size (width, height)
        save_path : str or Path, optional
            Path to save the visualization. If None, creates default path
        title : str, optional
            Custom title for the plot. If None, uses default
        
        Returns:
        --------
        Path
            Path to saved visualization file
        
        Raises:
        -------
        ValueError
            If top_n is invalid
        """
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1. Got {top_n}")
        
        if top_n > len(importance_results.importance_df):
            logger.warning(
                f"top_n ({top_n}) exceeds number of features "
                f"({len(importance_results.importance_df)}). Using all features."
            )
            top_n = len(importance_results.importance_df)
        
        # Get top N features
        top_features_df = importance_results.importance_df.head(top_n).copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        bars = ax.barh(
            range(len(top_features_df)),
            top_features_df['importance'].values,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.2
        )
        
        # Customize axes
        ax.set_yticks(range(len(top_features_df)))
        ax.set_yticklabels(top_features_df['feature'].values, fontsize=11)
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        
        # Set title
        if title is None:
            title = f'Top {top_n} Most Important Features\n({importance_results.method.title()} Method)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features_df.iterrows()):
            value = row['importance']
            ax.text(
                value + (max(top_features_df['importance']) * 0.01),
                i,
                f'{value:.4f}',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        # Invert y-axis to show highest importance at top
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = Path('visualizations') / f'feature_importance_top{top_n}.png'
        else:
            save_path = Path(save_path)
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Feature importance visualization saved to: {save_path}")
        
        return save_path
    
    def get_top_features(
        self,
        importance_results: FeatureImportanceResults,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N features as a formatted DataFrame.
        
        Parameters:
        -----------
        importance_results : FeatureImportanceResults
            Feature importance results
        top_n : int, default 10
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with top N features and their importance scores
        """
        top_df = importance_results.importance_df.head(top_n).copy()
        top_df = top_df[['feature', 'importance']].reset_index(drop=True)
        top_df.index = top_df.index + 1  # Start from 1 instead of 0
        top_df.index.name = 'Rank'
        
        return top_df
    
    def print_feature_importance_summary(
        self,
        importance_results: FeatureImportanceResults,
        top_n: int = 10
    ) -> None:
        """
        Print a formatted summary of top features.
        
        Parameters:
        -----------
        importance_results : FeatureImportanceResults
            Feature importance results
        top_n : int, default 10
            Number of top features to display
        """
        print("\n" + "=" * 80)
        print(f"FEATURE IMPORTANCE SUMMARY (Top {top_n})")
        print("=" * 80)
        print(f"Method: {importance_results.method.title()}")
        print(f"Total Features: {len(importance_results.importance_df)}")
        print("-" * 80)
        
        top_df = self.get_top_features(importance_results, top_n=top_n)
        print(top_df.to_string())
        
        print("\n" + "=" * 80)
    
    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_size: Optional[int] = 100,
        background_size: Optional[int] = 100
    ) -> Tuple[np.ndarray, Any]:
        """
        Compute SHAP values for the given data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features to explain
        sample_size : int, optional
            Number of samples to compute SHAP values for. If None, uses all samples.
            For large datasets, use a subset for efficiency.
        background_size : int, optional
            Number of background samples for TreeExplainer. If None, uses all training data.
            For large datasets, use a subset for efficiency.
        
        Returns:
        --------
        Tuple[np.ndarray, Any]
            SHAP values array and SHAP explainer object
        
        Raises:
        -------
        ImportError
            If SHAP is not installed
        ValueError
            If inputs are invalid
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )
        
        logger.info("Computing SHAP values...")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            feature_names = self.feature_names
        
        # Sample data if needed
        if sample_size is not None and len(X_array) > sample_size:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(len(X_array), sample_size, replace=False)
            X_sample = X_array[sample_indices]
            logger.info(f"Sampling {sample_size} instances from {len(X_array)} total")
        else:
            X_sample = X_array
            sample_indices = np.arange(len(X_array))
        
        # Determine explainer type based on model
        model_type = type(self.model).__name__.lower()
        
        try:
            # Tree-based models (Random Forest, XGBoost, LightGBM)
            if any(x in model_type for x in ['randomforest', 'xgboost', 'lgbm', 'lightgbm', 'gradientboosting']):
                logger.info("Using TreeExplainer for tree-based model")
                
                # Use background data for TreeExplainer (optional, but recommended)
                if background_size is not None and len(X_array) > background_size:
                    np.random.seed(self.random_state)
                    bg_indices = np.random.choice(len(X_array), background_size, replace=False)
                    X_background = X_array[bg_indices]
                else:
                    X_background = X_array[:min(100, len(X_array))]  # Use first 100 as background
                
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class output (take positive class for binary classification)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class for binary classification
            
            # Linear models (Logistic Regression)
            elif any(x in model_type for x in ['logistic', 'linear']):
                logger.info("Using LinearExplainer for linear model")
                
                # Use background data
                if background_size is not None and len(X_array) > background_size:
                    np.random.seed(self.random_state)
                    bg_indices = np.random.choice(len(X_array), background_size, replace=False)
                    X_background = X_array[bg_indices]
                else:
                    X_background = X_array[:min(100, len(X_array))]
                
                explainer = shap.LinearExplainer(self.model, X_background)
                shap_values = explainer.shap_values(X_sample)
            
            # Generic model (KernelExplainer - slower but works for any model)
            else:
                logger.info("Using KernelExplainer for generic model (this may be slow)")
                
                # Use background data
                if background_size is not None and len(X_array) > background_size:
                    np.random.seed(self.random_state)
                    bg_indices = np.random.choice(len(X_array), background_size, replace=False)
                    X_background = X_array[bg_indices]
                else:
                    X_background = X_array[:min(50, len(X_array))]  # Smaller background for KernelExplainer
                
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    X_background
                )
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            logger.info(f"✓ Computed SHAP values: shape {shap_values.shape}")
            
            # Store explainer and update feature names if needed
            self.shap_explainer = explainer
            if feature_names and len(feature_names) == shap_values.shape[1]:
                self.feature_names = feature_names
            
            return shap_values, explainer
        
        except Exception as e:
            error_msg = f"Error computing SHAP values: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate_shap_summary_plot(
        self,
        shap_values: np.ndarray,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> Path:
        """
        Generate SHAP summary plot (global feature importance).
        
        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP values array
        X : pd.DataFrame or np.ndarray
            Features corresponding to SHAP values
        max_display : int, default 10
            Maximum number of features to display
        save_path : str or Path, optional
            Path to save the plot. If None, creates default path
        title : str, optional
            Custom title for the plot
        
        Returns:
        --------
        Path
            Path to saved visualization file
        
        Raises:
        -------
        ImportError
            If SHAP is not installed
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )
        
        logger.info("Generating SHAP summary plot...")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            feature_names = self.feature_names
        
        # Ensure shapes match
        if len(X_array) != len(shap_values):
            raise ValueError(
                f"X and shap_values must have same length. "
                f"Got {len(X_array)} and {len(shap_values)}"
            )
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        
        # Use feature names if available
        if feature_names and len(feature_names) == shap_values.shape[1]:
            shap.summary_plot(
                shap_values,
                X_array,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                X_array,
                max_display=max_display,
                show=False
            )
        
        # Set title
        if title:
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            plt.title(
                f'SHAP Summary Plot (Top {max_display} Features)',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = Path('visualizations') / 'shap_summary_plot.png'
        else:
            save_path = Path(save_path)
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ SHAP summary plot saved to: {save_path}")
        
        return save_path
    
    def identify_case_studies(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        min_samples: int = 1
    ) -> List[CaseStudyInstance]:
        """
        Identify case study instances: TP, FP, FN from predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True labels
        min_samples : int, default 1
            Minimum number of samples to find for each case type
        
        Returns:
        --------
        List[CaseStudyInstance]
            List of case study instances with their metadata
        """
        logger.info("Identifying case study instances...")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        y_array = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Make predictions
        y_pred = self.model.predict(X_array)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_array)[:, 1]
        else:
            y_proba = np.zeros(len(y_pred))
        
        # Identify cases
        cases = []
        
        # True Positives (TP): predicted=1, actual=1
        tp_indices = np.where((y_pred == 1) & (y_array == 1))[0]
        if len(tp_indices) >= min_samples:
            for idx in tp_indices[:max(min_samples, 1)]:
                cases.append(CaseStudyInstance(
                    index=int(idx),
                    prediction=int(y_pred[idx]),
                    actual=int(y_array[idx]),
                    case_type='TP',
                    probability=float(y_proba[idx])
                ))
        
        # False Positives (FP): predicted=1, actual=0
        fp_indices = np.where((y_pred == 1) & (y_array == 0))[0]
        if len(fp_indices) >= min_samples:
            for idx in fp_indices[:max(min_samples, 1)]:
                cases.append(CaseStudyInstance(
                    index=int(idx),
                    prediction=int(y_pred[idx]),
                    actual=int(y_array[idx]),
                    case_type='FP',
                    probability=float(y_proba[idx])
                ))
        
        # False Negatives (FN): predicted=0, actual=1
        fn_indices = np.where((y_pred == 0) & (y_array == 1))[0]
        if len(fn_indices) >= min_samples:
            for idx in fn_indices[:max(min_samples, 1)]:
                cases.append(CaseStudyInstance(
                    index=int(idx),
                    prediction=int(y_pred[idx]),
                    actual=int(y_array[idx]),
                    case_type='FN',
                    probability=float(y_proba[idx])
                ))
        
        logger.info(f"✓ Identified {len(cases)} case study instances:")
        for case in cases:
            logger.info(f"  {case.case_type}: index={case.index}, prob={case.probability:.4f}")
        
        return cases
    
    def generate_shap_force_plot(
        self,
        shap_values: np.ndarray,
        X: Union[pd.DataFrame, np.ndarray],
        case_instance: CaseStudyInstance,
        save_path: Optional[Union[str, Path]] = None,
        plot_type: str = 'html'
    ) -> Path:
        """
        Generate SHAP force plot for a specific instance.
        
        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP values array
        X : pd.DataFrame or np.ndarray
            Features corresponding to SHAP values
        case_instance : CaseStudyInstance
            Case study instance to visualize
        save_path : str or Path, optional
            Path to save the plot. If None, creates default path
        plot_type : str, default 'html'
            Type of plot: 'html' or 'matplotlib'
        
        Returns:
        --------
        Path
            Path to saved visualization file
        
        Raises:
        -------
        ImportError
            If SHAP is not installed
        ValueError
            If case instance index is invalid
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )
        
        logger.info(f"Generating SHAP force plot for {case_instance.case_type} (index={case_instance.index})...")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            feature_names = self.feature_names
        
        # Validate index
        if case_instance.index >= len(X_array) or case_instance.index >= len(shap_values):
            raise ValueError(
                f"Case instance index {case_instance.index} is out of range. "
                f"X has {len(X_array)} samples, shap_values has {len(shap_values)} samples."
            )
        
        # Get instance data
        instance_shap = shap_values[case_instance.index]
        instance_features = X_array[case_instance.index]
        
        # Get base value (expected value)
        if self.shap_explainer is not None:
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                base_value = 0.0
        else:
            base_value = 0.0
        
        # Create save path
        if save_path is None:
            save_path = Path('visualizations') / f'shap_force_{case_instance.case_type}_idx{case_instance.index}.{plot_type}'
        else:
            save_path = Path(save_path)
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if plot_type == 'html':
            # HTML force plot (interactive)
            try:
                force_plot = shap.force_plot(
                    base_value,
                    instance_shap,
                    instance_features,
                    feature_names=feature_names if feature_names else None,
                    show=False,
                    matplotlib=False
                )
                
                # Save HTML
                shap.save_html(str(save_path), force_plot)
                logger.info(f"✓ SHAP force plot (HTML) saved to: {save_path}")
            
            except Exception as e:
                logger.warning(f"Could not create HTML force plot: {e}. Trying matplotlib version...")
                plot_type = 'matplotlib'
        
        if plot_type == 'matplotlib':
            # Matplotlib force plot (static)
            try:
                plt.figure(figsize=(12, 4))
                shap.force_plot(
                    base_value,
                    instance_shap,
                    instance_features,
                    feature_names=feature_names if feature_names else None,
                    show=False,
                    matplotlib=True
                )
                
                plt.title(
                    f'SHAP Force Plot - {case_instance.case_type}\n'
                    f'Index: {case_instance.index}, '
                    f'Predicted: {case_instance.prediction}, '
                    f'Actual: {case_instance.actual}, '
                    f'Probability: {case_instance.probability:.4f}',
                    fontsize=12,
                    fontweight='bold',
                    pad=10
                )
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                logger.info(f"✓ SHAP force plot (matplotlib) saved to: {save_path}")
            
            except Exception as e:
                error_msg = f"Error creating matplotlib force plot: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        return save_path

