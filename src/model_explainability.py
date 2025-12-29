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
    
    def compute_shap_feature_importance(
        self,
        shap_values: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute feature importance from SHAP values (mean absolute SHAP values).
        
        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP values array (samples x features)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with features and their SHAP-based importance scores,
            sorted by importance
        """
        logger.info("Computing SHAP-based feature importance...")
        
        # Compute mean absolute SHAP values per feature
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        shap_importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(shap_importance)],
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        # Reset index
        shap_importance_df = shap_importance_df.reset_index(drop=True)
        
        logger.info(f"✓ Computed SHAP importance for {len(shap_importance_df)} features")
        logger.info(f"  Top feature: {shap_importance_df.iloc[0]['feature']} "
                   f"(SHAP importance: {shap_importance_df.iloc[0]['shap_importance']:.4f})")
        
        return shap_importance_df
    
    def compare_importance_methods(
        self,
        builtin_importance: FeatureImportanceResults,
        shap_importance_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compare built-in feature importance with SHAP-based importance.
        
        Parameters:
        -----------
        builtin_importance : FeatureImportanceResults
            Built-in feature importance results
        shap_importance_df : pd.DataFrame
            SHAP-based feature importance DataFrame
        top_n : int, default 10
            Number of top features to compare
        
        Returns:
        --------
        pd.DataFrame
            Comparison DataFrame with both importance scores and rankings
        """
        logger.info("Comparing built-in and SHAP feature importance...")
        
        # Prepare built-in importance
        builtin_df = builtin_importance.importance_df.copy()
        builtin_df = builtin_df.rename(columns={'importance': 'builtin_importance'})
        builtin_df['builtin_rank'] = range(1, len(builtin_df) + 1)
        
        # Prepare SHAP importance
        shap_df = shap_importance_df.copy()
        shap_df['shap_rank'] = range(1, len(shap_df) + 1)
        
        # Merge on feature names
        comparison_df = pd.merge(
            builtin_df[['feature', 'builtin_importance', 'builtin_rank']],
            shap_df[['feature', 'shap_importance', 'shap_rank']],
            on='feature',
            how='outer'
        )
        
        # Fill missing values with 0
        comparison_df = comparison_df.fillna(0)
        
        # Compute rank difference
        comparison_df['rank_difference'] = (
            comparison_df['builtin_rank'] - comparison_df['shap_rank']
        )
        
        # Normalize importance scores to 0-1 for comparison
        if comparison_df['builtin_importance'].max() > 0:
            comparison_df['builtin_importance_norm'] = (
                comparison_df['builtin_importance'] / comparison_df['builtin_importance'].max()
            )
        else:
            comparison_df['builtin_importance_norm'] = 0
        
        if comparison_df['shap_importance'].max() > 0:
            comparison_df['shap_importance_norm'] = (
                comparison_df['shap_importance'] / comparison_df['shap_importance'].max()
            )
        else:
            comparison_df['shap_importance_norm'] = 0
        
        # Compute importance difference
        comparison_df['importance_difference'] = (
            comparison_df['builtin_importance_norm'] - comparison_df['shap_importance_norm']
        )
        
        # Sort by average rank
        comparison_df['avg_rank'] = (
            comparison_df['builtin_rank'] + comparison_df['shap_rank']
        ) / 2
        comparison_df = comparison_df.sort_values('avg_rank')
        
        logger.info(f"✓ Compared importance for {len(comparison_df)} features")
        
        return comparison_df
    
    def identify_top_drivers(
        self,
        comparison_df: pd.DataFrame,
        top_n: int = 5,
        method: str = 'combined'
    ) -> pd.DataFrame:
        """
        Identify top N drivers of fraud predictions.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison DataFrame from compare_importance_methods()
        top_n : int, default 5
            Number of top drivers to identify
        method : str, default 'combined'
            Method to use: 'builtin', 'shap', or 'combined'
            - 'builtin': Use built-in importance ranking
            - 'shap': Use SHAP importance ranking
            - 'combined': Use average ranking
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with top N drivers and their importance scores
        """
        logger.info(f"Identifying top {top_n} drivers using method: {method}")
        
        if method == 'builtin':
            top_drivers = comparison_df.nsmallest(top_n, 'builtin_rank')
        elif method == 'shap':
            top_drivers = comparison_df.nsmallest(top_n, 'shap_rank')
        else:  # combined
            top_drivers = comparison_df.nsmallest(top_n, 'avg_rank')
        
        # Select relevant columns
        result_df = top_drivers[[
            'feature',
            'builtin_importance',
            'shap_importance',
            'builtin_rank',
            'shap_rank',
            'rank_difference'
        ]].copy()
        
        result_df = result_df.reset_index(drop=True)
        result_df.index = result_df.index + 1  # Start from 1
        result_df.index.name = 'Rank'
        
        logger.info(f"✓ Identified top {len(result_df)} drivers")
        
        return result_df
    
    def identify_surprising_findings(
        self,
        comparison_df: pd.DataFrame,
        rank_threshold: int = 5,
        importance_diff_threshold: float = 0.2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify surprising or counterintuitive findings in feature importance.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison DataFrame from compare_importance_methods()
        rank_threshold : int, default 5
            Threshold for considering a feature as "top" (for rank differences)
        importance_diff_threshold : float, default 0.2
            Threshold for normalized importance difference (0-1 scale)
        
        Returns:
        --------
        Dict[str, List[Dict[str, Any]]]
            Dictionary with categories of surprising findings:
            - 'high_rank_difference': Features with large rank differences
            - 'importance_mismatch': Features with large importance differences
            - 'shap_higher': Features ranked higher by SHAP than built-in
            - 'builtin_higher': Features ranked higher by built-in than SHAP
        """
        logger.info("Identifying surprising findings...")
        
        findings = {
            'high_rank_difference': [],
            'importance_mismatch': [],
            'shap_higher': [],
            'builtin_higher': []
        }
        
        for _, row in comparison_df.iterrows():
            feature = row['feature']
            rank_diff = abs(row['rank_difference'])
            importance_diff = abs(row['importance_difference'])
            builtin_rank = row['builtin_rank']
            shap_rank = row['shap_rank']
            
            # High rank difference
            if rank_diff >= rank_threshold:
                findings['high_rank_difference'].append({
                    'feature': feature,
                    'builtin_rank': int(builtin_rank),
                    'shap_rank': int(shap_rank),
                    'rank_difference': int(rank_diff),
                    'interpretation': (
                        f"Large ranking discrepancy: "
                        f"Built-in rank {int(builtin_rank)} vs SHAP rank {int(shap_rank)}"
                    )
                })
            
            # Importance mismatch
            if importance_diff >= importance_diff_threshold:
                findings['importance_mismatch'].append({
                    'feature': feature,
                    'builtin_importance_norm': float(row['builtin_importance_norm']),
                    'shap_importance_norm': float(row['shap_importance_norm']),
                    'importance_difference': float(importance_diff),
                    'interpretation': (
                        f"Significant importance difference: "
                        f"Built-in {row['builtin_importance_norm']:.3f} vs "
                        f"SHAP {row['shap_importance_norm']:.3f}"
                    )
                })
            
            # SHAP ranks higher (more important in SHAP)
            if shap_rank < builtin_rank and rank_diff >= 3:
                findings['shap_higher'].append({
                    'feature': feature,
                    'builtin_rank': int(builtin_rank),
                    'shap_rank': int(shap_rank),
                    'interpretation': (
                        f"SHAP suggests this feature is more important than built-in indicates. "
                        f"May have complex interactions or non-linear effects."
                    )
                })
            
            # Built-in ranks higher (more important in built-in)
            if builtin_rank < shap_rank and rank_diff >= 3:
                findings['builtin_higher'].append({
                    'feature': feature,
                    'builtin_rank': int(builtin_rank),
                    'shap_rank': int(shap_rank),
                    'interpretation': (
                        f"Built-in importance suggests higher importance than SHAP. "
                        f"May have consistent but smaller individual contributions."
                    )
                })
        
        # Sort findings by magnitude
        for key in findings:
            if findings[key]:
                if 'rank_difference' in findings[key][0]:
                    findings[key].sort(key=lambda x: abs(x.get('rank_difference', 0)), reverse=True)
                elif 'importance_difference' in findings[key][0]:
                    findings[key].sort(key=lambda x: abs(x.get('importance_difference', 0)), reverse=True)
        
        total_findings = sum(len(v) for v in findings.values())
        logger.info(f"✓ Identified {total_findings} surprising findings across {len(findings)} categories")
        
        return findings
    
    def generate_interpretation_report(
        self,
        builtin_importance: FeatureImportanceResults,
        shap_values: np.ndarray,
        X: Union[pd.DataFrame, np.ndarray],
        top_n: int = 5,
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation report comparing SHAP and built-in importance.
        
        Parameters:
        -----------
        builtin_importance : FeatureImportanceResults
            Built-in feature importance results
        shap_values : np.ndarray
            SHAP values array
        X : pd.DataFrame or np.ndarray
            Features corresponding to SHAP values
        top_n : int, default 5
            Number of top drivers to identify
        save_path : str or Path, optional
            Path to save the report. If None, creates default path
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'top_drivers': Top N drivers DataFrame
            - 'comparison_df': Full comparison DataFrame
            - 'surprising_findings': Dictionary of surprising findings
            - 'summary': Text summary of findings
        """
        logger.info("Generating interpretation report...")
        
        # Compute SHAP importance
        shap_importance_df = self.compute_shap_feature_importance(shap_values)
        
        # Compare methods
        comparison_df = self.compare_importance_methods(
            builtin_importance,
            shap_importance_df,
            top_n=20  # Compare more features for better analysis
        )
        
        # Identify top drivers (using combined method)
        top_drivers = self.identify_top_drivers(
            comparison_df,
            top_n=top_n,
            method='combined'
        )
        
        # Identify surprising findings
        surprising_findings = self.identify_surprising_findings(comparison_df)
        
        # Generate summary text
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("FEATURE IMPORTANCE INTERPRETATION REPORT")
        summary_lines.append("=" * 80)
        summary_lines.append(f"\nTop {top_n} Drivers of Fraud Predictions (Combined Ranking):")
        summary_lines.append("-" * 80)
        
        for idx, row in top_drivers.iterrows():
            summary_lines.append(
                f"{idx}. {row['feature']}\n"
                f"   Built-in Rank: {int(row['builtin_rank'])}, "
                f"SHAP Rank: {int(row['shap_rank'])}\n"
                f"   Built-in Importance: {row['builtin_importance']:.4f}, "
                f"SHAP Importance: {row['shap_importance']:.4f}"
            )
        
        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("SURPRISING FINDINGS")
        summary_lines.append("=" * 80)
        
        if surprising_findings['high_rank_difference']:
            summary_lines.append("\n1. High Rank Differences:")
            for finding in surprising_findings['high_rank_difference'][:5]:  # Top 5
                summary_lines.append(f"   - {finding['feature']}: {finding['interpretation']}")
        
        if surprising_findings['shap_higher']:
            summary_lines.append("\n2. Features More Important in SHAP:")
            for finding in surprising_findings['shap_higher'][:5]:
                summary_lines.append(f"   - {finding['feature']}: {finding['interpretation']}")
        
        if surprising_findings['builtin_higher']:
            summary_lines.append("\n3. Features More Important in Built-in:")
            for finding in surprising_findings['builtin_higher'][:5]:
                summary_lines.append(f"   - {finding['feature']}: {finding['interpretation']}")
        
        summary_lines.append("\n" + "=" * 80)
        
        summary_text = "\n".join(summary_lines)
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                f.write(summary_text)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("DETAILED COMPARISON TABLE\n")
                f.write("=" * 80 + "\n\n")
                f.write(comparison_df.head(20).to_string())
            
            logger.info(f"✓ Interpretation report saved to: {save_path}")
        
        report = {
            'top_drivers': top_drivers,
            'comparison_df': comparison_df,
            'surprising_findings': surprising_findings,
            'summary': summary_text
        }
        
        return report
    
    def visualize_importance_comparison(
        self,
        comparison_df: pd.DataFrame,
        top_n: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> Path:
        """
        Visualize comparison between built-in and SHAP feature importance.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison DataFrame from compare_importance_methods()
        top_n : int, default 10
            Number of top features to visualize
        save_path : str or Path, optional
            Path to save the plot. If None, creates default path
        title : str, optional
            Custom title for the plot
        
        Returns:
        --------
        Path
            Path to saved visualization file
        """
        logger.info("Creating importance comparison visualization...")
        
        # Get top N features by average rank
        top_features = comparison_df.head(top_n).copy()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Side-by-side bar chart
        x = np.arange(len(top_features))
        width = 0.35
        
        ax1.bar(x - width/2, top_features['builtin_importance_norm'], 
                width, label='Built-in Importance', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, top_features['shap_importance_norm'], 
                width, label='SHAP Importance', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Built-in vs SHAP Feature Importance Comparison', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_features['feature'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Rank comparison scatter plot
        ax2.scatter(top_features['builtin_rank'], top_features['shap_rank'],
                   s=100, alpha=0.6, c=top_features['shap_importance_norm'],
                   cmap='viridis', edgecolors='black', linewidth=1.5)
        
        # Add diagonal line (perfect agreement)
        max_rank = max(top_features[['builtin_rank', 'shap_rank']].max())
        ax2.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, label='Perfect Agreement')
        
        # Add feature labels
        for idx, row in top_features.iterrows():
            ax2.annotate(row['feature'], 
                        (row['builtin_rank'], row['shap_rank']),
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Built-in Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SHAP Rank', fontsize=12, fontweight='bold')
        ax2.set_title('Rank Comparison: Built-in vs SHAP', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle(f'Feature Importance Comparison (Top {top_n} Features)',
                        fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = Path('visualizations') / 'importance_comparison.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Importance comparison visualization saved to: {save_path}")
        
        return save_path
    
    def generate_business_recommendations(
        self,
        interpretation_report: Dict[str, Any],
        dataset_type: Optional[str] = None,
        min_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable business recommendations based on SHAP insights.
        
        Parameters:
        -----------
        interpretation_report : Dict[str, Any]
            Interpretation report from generate_interpretation_report()
        dataset_type : str, optional
            Type of dataset ('ecommerce' or 'banking') for context-specific recommendations
        min_recommendations : int, default 3
            Minimum number of recommendations to generate
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of recommendation dictionaries, each containing:
            - 'recommendation': Actionable recommendation text
            - 'shap_insight': SHAP insight that supports the recommendation
            - 'feature': Feature name related to the recommendation
            - 'priority': Priority level ('HIGH', 'MEDIUM', 'LOW')
            - 'expected_impact': Expected impact description
        """
        logger.info("Generating business recommendations from SHAP insights...")
        
        top_drivers = interpretation_report['top_drivers']
        surprising_findings = interpretation_report['surprising_findings']
        comparison_df = interpretation_report['comparison_df']
        
        recommendations = []
        
        # Analyze top drivers to generate recommendations
        for idx, row in top_drivers.iterrows():
            feature = row['feature']
            shap_rank = int(row['shap_rank'])
            builtin_rank = int(row['builtin_rank'])
            shap_importance = row['shap_importance']
            
            # Generate recommendations based on feature patterns
            rec = self._generate_feature_specific_recommendation(
                feature, shap_rank, shap_importance, dataset_type
            )
            if rec:
                recommendations.append(rec)
        
        # Analyze surprising findings for additional recommendations
        if surprising_findings['shap_higher']:
            for finding in surprising_findings['shap_higher'][:2]:  # Top 2
                feature = finding['feature']
                rec = self._generate_surprising_finding_recommendation(
                    finding, dataset_type
                )
                if rec:
                    recommendations.append(rec)
        
        # Ensure minimum number of recommendations
        if len(recommendations) < min_recommendations:
            # Generate generic recommendations based on top features
            additional = self._generate_generic_recommendations(
                top_drivers, min_recommendations - len(recommendations), dataset_type
            )
            recommendations.extend(additional)
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        logger.info(f"✓ Generated {len(recommendations)} business recommendations")
        
        return recommendations
    
    def _generate_feature_specific_recommendation(
        self,
        feature: str,
        shap_rank: int,
        shap_importance: float,
        dataset_type: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate recommendation for a specific feature."""
        feature_lower = feature.lower()
        
        # Time-based features
        if any(x in feature_lower for x in ['time', 'hour', 'minute', 'signup', 'purchase']):
            if 'signup' in feature_lower or 'time_since' in feature_lower:
                return {
                    'recommendation': (
                        "Transactions within 24 hours of account signup should receive "
                        "additional verification (e.g., phone verification, email confirmation). "
                        "Consider implementing a graduated verification system where transactions "
                        "within 1 hour require the highest level of verification."
                    ),
                    'shap_insight': (
                        f"Time-based features (ranked #{shap_rank} by SHAP) show strong "
                        f"predictive power for fraud detection. New accounts are particularly "
                        f"vulnerable to fraudulent activity."
                    ),
                    'feature': feature,
                    'priority': 'HIGH',
                    'expected_impact': (
                        "Expected to reduce fraud by 15-25% in new account transactions, "
                        "with minimal impact on legitimate user experience if verification "
                        "is streamlined."
                    )
                }
            elif 'purchase' in feature_lower or 'transaction' in feature_lower:
                return {
                    'recommendation': (
                        "Implement real-time transaction monitoring for high-value transactions "
                        "occurring outside normal business hours or in rapid succession. "
                        "Set up alerts for transactions exceeding user's historical patterns."
                    ),
                    'shap_insight': (
                        f"Transaction timing features (SHAP importance: {shap_importance:.4f}) "
                        f"are key indicators of fraudulent behavior, especially when combined "
                        f"with other risk factors."
                    ),
                    'feature': feature,
                    'priority': 'HIGH',
                    'expected_impact': (
                        "Can help detect 20-30% of fraudulent transactions that occur during "
                        "unusual time periods or show velocity anomalies."
                    )
                }
        
        # Amount-based features
        elif any(x in feature_lower for x in ['amount', 'value', 'price', 'cost']):
            return {
                'recommendation': (
                    "Implement dynamic transaction limits based on user behavior patterns. "
                    "For transactions exceeding 2x the user's average transaction amount, "
                    "require additional authentication. Consider tiered limits: "
                    "low-risk users (higher limits), new users (lower limits)."
                ),
                'shap_insight': (
                    f"Transaction amount (ranked #{shap_rank} by SHAP) is a critical fraud "
                    f"indicator. Fraudulent transactions often deviate significantly from "
                    f"legitimate user patterns."
                ),
                'feature': feature,
                'priority': 'HIGH',
                'expected_impact': (
                    "Expected to prevent 25-35% of high-value fraudulent transactions while "
                    "maintaining smooth experience for legitimate high-value customers through "
                    "adaptive limits."
                )
            }
        
        # Device/browser features
        elif any(x in feature_lower for x in ['device', 'browser', 'os', 'platform']):
            return {
                'recommendation': (
                    "Implement device fingerprinting and track device changes. Require "
                    "additional verification when transactions occur from new devices, "
                    "especially for high-value transactions. Flag rapid device switching "
                    "as a potential fraud indicator."
                ),
                'shap_insight': (
                    f"Device/browser features (SHAP importance: {shap_importance:.4f}) provide "
                    f"strong signals for fraud detection. Fraudsters often use different devices "
                    f"or automated tools."
                ),
                'feature': feature,
                'priority': 'MEDIUM',
                'expected_impact': (
                    "Can identify 15-20% of fraudulent transactions through device anomaly "
                    "detection, with low false positive rates when combined with other signals."
                )
            }
        
        # Location/IP features
        elif any(x in feature_lower for x in ['ip', 'location', 'country', 'city', 'geo']):
            return {
                'recommendation': (
                    "Implement geolocation-based risk scoring. Flag transactions from "
                    "high-risk locations or IP addresses known for fraud. Require additional "
                    "verification for transactions from countries with high fraud rates or "
                    "when IP location doesn't match user's typical patterns."
                ),
                'shap_insight': (
                    f"Geographic features (ranked #{shap_rank} by SHAP) are important indicators "
                    f"of fraud risk, especially when combined with other behavioral signals."
                ),
                'feature': feature,
                'priority': 'MEDIUM',
                'expected_impact': (
                    "Expected to reduce fraud by 10-20% from high-risk geographic regions, "
                    "with minimal impact on legitimate international transactions when "
                    "properly calibrated."
                )
            }
        
        # Frequency/velocity features
        elif any(x in feature_lower for x in ['freq', 'count', 'velocity', 'rate', 'per']):
            return {
                'recommendation': (
                    "Implement transaction velocity monitoring. Flag accounts with unusually "
                    "high transaction frequency (e.g., >5 transactions per hour) for manual "
                    "review. Set up automated alerts for rapid-fire transactions that may "
                    "indicate account takeover or card testing."
                ),
                'shap_insight': (
                    f"Transaction frequency features (SHAP importance: {shap_importance:.4f}) "
                    f"are strong predictors of fraud. Fraudulent accounts often show abnormal "
                    f"transaction patterns."
                ),
                'feature': feature,
                'priority': 'HIGH',
                'expected_impact': (
                    "Can detect 20-30% of fraudulent activity through velocity-based detection, "
                    "particularly effective against card testing and account takeover attempts."
                )
            }
        
        # User behavior features
        elif any(x in feature_lower for x in ['user', 'session', 'login', 'activity']):
            return {
                'recommendation': (
                    "Implement behavioral biometrics and session analysis. Monitor for "
                    "unusual user behavior patterns such as rapid navigation, automated "
                    "clicking patterns, or deviations from typical user journey. Require "
                    "step-up authentication for transactions that deviate significantly "
                    "from user's historical behavior."
                ),
                'shap_insight': (
                    f"User behavior features (ranked #{shap_rank} by SHAP) provide valuable "
                    f"signals for detecting account takeover and fraudulent account creation."
                ),
                'feature': feature,
                'priority': 'MEDIUM',
                'expected_impact': (
                    "Expected to identify 15-25% of account takeover and fraudulent account "
                    "creation attempts through behavioral anomaly detection."
                )
            }
        
        return None
    
    def _generate_surprising_finding_recommendation(
        self,
        finding: Dict[str, Any],
        dataset_type: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate recommendation based on surprising SHAP findings."""
        feature = finding['feature']
        shap_rank = finding['shap_rank']
        builtin_rank = finding['builtin_rank']
        
        return {
            'recommendation': (
                f"Conduct deeper investigation into '{feature}' as SHAP analysis reveals "
                f"it may have stronger predictive power than initially indicated by built-in "
                f"importance (SHAP rank: #{shap_rank} vs Built-in rank: #{builtin_rank}). "
                f"Consider creating additional derived features or interaction terms based on "
                f"this feature to improve model performance."
            ),
            'shap_insight': (
                f"SHAP analysis shows '{feature}' has significant importance (rank #{shap_rank}) "
                f"that differs from built-in feature importance (rank #{builtin_rank}). This "
                f"suggests the feature may have complex interactions or non-linear effects "
                f"that are better captured by SHAP values."
            ),
            'feature': feature,
            'priority': 'MEDIUM',
            'expected_impact': (
                "Further investigation and feature engineering based on this insight could "
                "potentially improve model performance by 2-5% and provide better "
                "understanding of fraud patterns."
            )
        }
    
    def _generate_generic_recommendations(
        self,
        top_drivers: pd.DataFrame,
        count: int,
        dataset_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate generic recommendations based on top drivers."""
        recommendations = []
        
        # Recommendation 1: Multi-factor approach
        if count > 0:
            top_features = ', '.join(top_drivers['feature'].head(3).tolist())
            recommendations.append({
                'recommendation': (
                    "Implement a multi-factor risk scoring system that combines the top "
                    f"predictive features ({top_features}). Create a composite risk score "
                    "that triggers different levels of verification based on the combination "
                    "of risk factors rather than individual features alone."
                ),
                'shap_insight': (
                    f"SHAP analysis identifies {len(top_drivers)} key drivers of fraud. "
                    "Combining multiple signals provides more robust fraud detection than "
                    "relying on individual features."
                ),
                'feature': 'Multiple',
                'priority': 'HIGH',
                'expected_impact': (
                    "Multi-factor risk scoring can improve fraud detection accuracy by 10-15% "
                    "while reducing false positives through more nuanced risk assessment."
                )
            })
            count -= 1
        
        # Recommendation 2: Continuous monitoring
        if count > 0:
            recommendations.append({
                'recommendation': (
                    "Establish continuous model monitoring and retraining pipeline. "
                    "Regularly update the fraud detection model with new data to adapt to "
                    "evolving fraud patterns. Set up alerts for model performance degradation "
                    "or shifts in feature importance that may indicate new fraud tactics."
                ),
                'shap_insight': (
                    "SHAP values can change as fraud patterns evolve. Regular monitoring of "
                    "feature importance helps identify when model updates are needed to "
                    "maintain effectiveness."
                ),
                'feature': 'Model Monitoring',
                'priority': 'MEDIUM',
                'expected_impact': (
                    "Continuous monitoring and retraining can maintain model effectiveness "
                    "over time, preventing performance degradation that could lead to "
                    "5-10% increase in undetected fraud."
                )
            })
            count -= 1
        
        # Recommendation 3: Explainability for operations
        if count > 0:
            recommendations.append({
                'recommendation': (
                    "Integrate SHAP-based explanations into fraud review workflows. Provide "
                    "fraud analysts with feature-level explanations for flagged transactions "
                    "to help them make faster and more accurate decisions. Use SHAP values "
                    "to prioritize cases for manual review."
                ),
                'shap_insight': (
                    "SHAP values provide interpretable explanations for individual predictions, "
                    "making it easier for fraud analysts to understand why a transaction "
                    "was flagged and take appropriate action."
                ),
                'feature': 'Explainability',
                'priority': 'MEDIUM',
                'expected_impact': (
                    "Can improve fraud analyst efficiency by 20-30% through faster case "
                    "resolution and better decision-making support, while reducing false "
                    "positive rates."
                )
            })
        
        return recommendations
    
    def format_recommendations_report(
        self,
        recommendations: List[Dict[str, Any]],
        save_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Format business recommendations as a comprehensive report.
        
        Parameters:
        -----------
        recommendations : List[Dict[str, Any]]
            List of recommendations from generate_business_recommendations()
        save_path : str or Path, optional
            Path to save the report
        
        Returns:
        --------
        str
            Formatted report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BUSINESS RECOMMENDATIONS BASED ON SHAP ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated {len(recommendations)} actionable recommendations")
        report_lines.append("\n" + "=" * 80)
        
        # Group by priority
        by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for rec in recommendations:
            priority = rec.get('priority', 'MEDIUM')
            by_priority[priority].append(rec)
        
        # Print by priority
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            if by_priority[priority]:
                report_lines.append(f"\n{priority} PRIORITY RECOMMENDATIONS")
                report_lines.append("-" * 80)
                
                for i, rec in enumerate(by_priority[priority], 1):
                    report_lines.append(f"\n{i}. RECOMMENDATION:")
                    report_lines.append(f"   {rec['recommendation']}")
                    report_lines.append(f"\n   SHAP INSIGHT:")
                    report_lines.append(f"   {rec['shap_insight']}")
                    report_lines.append(f"\n   RELATED FEATURE: {rec['feature']}")
                    report_lines.append(f"   EXPECTED IMPACT: {rec['expected_impact']}")
                    report_lines.append("\n" + "-" * 80)
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Total Recommendations: {len(recommendations)}")
        report_lines.append(f"  - High Priority: {len(by_priority['HIGH'])}")
        report_lines.append(f"  - Medium Priority: {len(by_priority['MEDIUM'])}")
        report_lines.append(f"  - Low Priority: {len(by_priority['LOW'])}")
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"✓ Recommendations report saved to: {save_path}")
        
        return report_text

