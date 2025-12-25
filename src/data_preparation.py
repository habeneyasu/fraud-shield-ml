"""
Data Preparation Module

This module provides a professional, reusable class-based approach for preparing
data for machine learning model training. It handles feature-target separation and
stratified train-test splitting with comprehensive validation and error handling.

Designed for fraud detection datasets:
- E-commerce: Fraud_Data.csv (target: 'class')
- Banking: creditcard.csv (target: 'Class')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSplitResult:
    """
    Data class to hold the results of data splitting.
    
    Attributes:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_train : pd.Series or np.ndarray
        Training target variable
    y_test : pd.Series or np.ndarray
        Test target variable
    train_size : int
        Number of samples in training set
    test_size : int
        Number of samples in test set
    train_class_distribution : Dict[str, int]
        Class distribution in training set
    test_class_distribution : Dict[str, int]
        Class distribution in test set
    """
    X_train: Union[pd.DataFrame, np.ndarray]
    X_test: Union[pd.DataFrame, np.ndarray]
    y_train: Union[pd.Series, np.ndarray]
    y_test: Union[pd.Series, np.ndarray]
    train_size: int
    test_size: int
    train_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]


class DataPreparation:
    """
    Professional data preparation class for fraud detection model training.
    
    This class provides a reusable, object-oriented approach to:
    - Separate features from target variables
    - Perform stratified train-test splitting
    - Validate data integrity
    - Handle multiple dataset types (e-commerce and banking)
    
    Attributes:
    -----------
    dataset_type : str
        Type of dataset: 'ecommerce' or 'banking'
    target_column : str
        Name of the target column ('class' or 'Class')
    exclude_columns : List[str]
        Columns to exclude from features (e.g., IDs, timestamps)
    random_state : int
        Random seed for reproducibility
    test_size : float
        Proportion of data for testing (default: 0.2)
    
    Example:
    --------
    >>> from src.data_preparation import DataPreparation
    >>> 
    >>> # For e-commerce dataset
    >>> prep = DataPreparation(
    ...     dataset_type='ecommerce',
    ...     exclude_columns=['user_id', 'signup_time', 'purchase_time', 'ip_address']
    ... )
    >>> split_result = prep.prepare_and_split(df)
    >>> 
    >>> # Access results
    >>> X_train = split_result.X_train
    >>> y_train = split_result.y_train
    """
    
    # Dataset configuration constants
    DATASET_TYPES = {
        'ecommerce': {
            'target_column': 'class',
            'default_exclude': ['user_id', 'signup_time', 'purchase_time', 'ip_address']
        },
        'banking': {
            'target_column': 'Class',
            'default_exclude': ['Time']  # Time can be excluded or kept as feature
        }
    }
    
    def __init__(
        self,
        dataset_type: str = 'ecommerce',
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ):
        """
        Initialize DataPreparation instance.
        
        Parameters:
        -----------
        dataset_type : str, default 'ecommerce'
            Type of dataset: 'ecommerce' or 'banking'
        target_column : str, optional
            Name of target column. If None, uses default for dataset_type
        exclude_columns : List[str], optional
            Columns to exclude from features. If None, uses defaults for dataset_type
        test_size : float, default 0.2
            Proportion of dataset to include in test split (0 < test_size < 1)
        random_state : int, default 42
            Random seed for reproducibility
        stratify : bool, default True
            Whether to use stratified splitting (preserves class distribution)
        
        Raises:
        -------
        ValueError
            If dataset_type is invalid or parameters are out of range
        """
        # Validate dataset_type
        if dataset_type not in self.DATASET_TYPES:
            raise ValueError(
                f"Invalid dataset_type: {dataset_type}. "
                f"Must be one of: {list(self.DATASET_TYPES.keys())}"
            )
        
        # Validate test_size
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")
        
        # Set attributes
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        
        # Set target column
        if target_column is None:
            self.target_column = self.DATASET_TYPES[dataset_type]['target_column']
        else:
            self.target_column = target_column
        
        # Set exclude columns
        if exclude_columns is None:
            self.exclude_columns = self.DATASET_TYPES[dataset_type]['default_exclude'].copy()
        else:
            self.exclude_columns = exclude_columns.copy()
        
        logger.info(f"Initialized DataPreparation for {dataset_type} dataset")
        logger.info(f"  Target column: {self.target_column}")
        logger.info(f"  Exclude columns: {self.exclude_columns}")
        logger.info(f"  Test size: {self.test_size}, Random state: {self.random_state}")
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe meets requirements for preparation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to validate
        
        Returns:
        --------
        bool
            True if validation passes
        
        Raises:
        -------
        ValueError
            If validation fails
        """
        if df is None:
            raise ValueError("Dataframe cannot be None")
        
        if df.empty:
            raise ValueError("Dataframe is empty")
        
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Check for missing values in target
        missing_target = df[self.target_column].isnull().sum()
        if missing_target > 0:
            logger.warning(
                f"Found {missing_target} missing values in target column '{self.target_column}'. "
                "These will need to be handled before splitting."
            )
        
        # Validate target is binary
        unique_values = df[self.target_column].dropna().unique()
        if len(unique_values) > 2:
            logger.warning(
                f"Target column has {len(unique_values)} unique values: {unique_values}. "
                "Expected binary classification (0/1)."
            )
        
        logger.info("Dataframe validation passed")
        return True
    
    def separate_features_target(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features from target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and target
        feature_columns : List[str], optional
            Specific columns to use as features. If None, uses all columns except
            target and excluded columns
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Features (X) and target (y)
        
        Raises:
        -------
        ValueError
            If dataframe is invalid or target column is missing
        """
        # Validate dataframe
        self.validate_dataframe(df)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Determine feature columns
        if feature_columns is None:
            # Use all columns except target and excluded columns
            feature_columns = [
                col for col in df.columns
                if col != self.target_column and col not in self.exclude_columns
            ]
        else:
            # Validate that specified columns exist
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found in dataframe: {missing_cols}")
            
            # Ensure target is not in feature columns
            if self.target_column in feature_columns:
                feature_columns.remove(self.target_column)
        
        if not feature_columns:
            raise ValueError(
                "No feature columns available. Check exclude_columns and feature_columns."
            )
        
        # Separate features and target
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Log information
        logger.info(f"Separated features and target")
        logger.info(f"  Features: {len(feature_columns)} columns")
        logger.info(f"  Feature columns: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")
        logger.info(f"  Target: {self.target_column}")
        logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        return_distribution: bool = True
    ) -> DataSplitResult:
        """
        Perform stratified train-test split with validation.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            Target variable
        return_distribution : bool, default True
            Whether to calculate and return class distributions
        
        Returns:
        --------
        DataSplitResult
            Dataclass containing split results and metadata
        
        Raises:
        -------
        ValueError
            If inputs are invalid
        """
        # Validate inputs
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        # Convert to numpy arrays for consistency (preserve DataFrame if input is DataFrame)
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            X_columns = X.columns.tolist()
            X_index = X.index
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got {len(X)} and {len(y)}"
            )
        
        # Check if stratification is possible
        stratify_param = None
        if self.stratify:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = class_counts.min()
            
            if min_class_count < 2:
                logger.warning(
                    f"Minimum class count ({min_class_count}) is less than 2. "
                    "Cannot use stratified split. Using non-stratified split."
                )
                stratify_param = None
            else:
                stratify_param = y
        
        logger.info(
            f"Splitting data: test_size={self.test_size}, "
            f"random_state={self.random_state}, "
            f"stratified={stratify_param is not None}"
        )
        
        # Perform split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_param
            )
        except Exception as e:
            error_msg = f"Error during train_test_split: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            X_train = pd.DataFrame(X_train, columns=X_columns)
            X_test = pd.DataFrame(X_test, columns=X_columns)
        
        # Calculate class distributions if requested
        train_class_dist = {}
        test_class_dist = {}
        if return_distribution:
            train_unique, train_counts = np.unique(y_train, return_counts=True)
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            
            train_class_dist = dict(zip(
                [f"Class_{int(c)}" for c in train_unique],
                train_counts.tolist()
            ))
            test_class_dist = dict(zip(
                [f"Class_{int(c)}" for c in test_unique],
                test_counts.tolist()
            ))
        
        logger.info(f"Split completed:")
        logger.info(f"  Train set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        if train_class_dist:
            logger.info(f"  Train class distribution: {train_class_dist}")
            logger.info(f"  Test class distribution: {test_class_dist}")
        
        # Create result object
        result = DataSplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_size=len(X_train),
            test_size=len(X_test),
            train_class_distribution=train_class_dist,
            test_class_distribution=test_class_dist
        )
        
        return result
    
    def prepare_and_split(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> DataSplitResult:
        """
        Complete data preparation pipeline: separate features/target and split.
        
        This is a convenience method that combines separate_features_target() and
        split_data() for a complete workflow.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and target
        feature_columns : List[str], optional
            Specific columns to use as features. If None, uses all columns except
            target and excluded columns
        
        Returns:
        --------
        DataSplitResult
            Dataclass containing split results and metadata
        
        Example:
        --------
        >>> prep = DataPreparation(dataset_type='ecommerce')
        >>> result = prep.prepare_and_split(df)
        >>> X_train, X_test = result.X_train, result.X_test
        >>> y_train, y_test = result.y_train, result.y_test
        """
        logger.info("Starting complete data preparation pipeline")
        
        # Step 1: Separate features and target
        X, y = self.separate_features_target(df, feature_columns=feature_columns)
        
        # Step 2: Split data
        result = self.split_data(X, y, return_distribution=True)
        
        logger.info("Data preparation pipeline completed successfully")
        
        return result
    
    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about features in the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing feature information
        """
        self.validate_dataframe(df)
        
        # Determine feature columns
        feature_columns = [
            col for col in df.columns
            if col != self.target_column and col not in self.exclude_columns
        ]
        
        info = {
            'total_columns': len(df.columns),
            'target_column': self.target_column,
            'excluded_columns': self.exclude_columns,
            'feature_columns': feature_columns,
            'num_features': len(feature_columns),
            'feature_types': df[feature_columns].dtypes.value_counts().to_dict() if feature_columns else {}
        }
        
        return info

