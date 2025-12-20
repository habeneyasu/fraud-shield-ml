"""
Data Preprocessing Module

This module provides reusable functions for data preprocessing with
validation checks and error handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime, timedelta

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for resampling (handle import errors gracefully)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError as e:
    IMBLEARN_AVAILABLE = False
    logger.warning(f"imbalanced-learn not available: {e}. Resampling features will be disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    columns: Optional[List[str]] = None,
    fill_value: Optional[Union[int, float, str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in a dataframe with validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str, default 'drop'
        Strategy to handle missing values: 'drop', 'fill', or 'forward_fill'
    columns : list, optional
        Specific columns to process. If None, processes all columns
    fill_value : int, float, or str, optional
        Value to use for filling when strategy is 'fill'
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    
    Raises:
    -------
    ValueError
        If strategy is invalid or dataframe is invalid
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        df = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        missing_before = df[columns].isnull().sum().sum()
        logger.info(f"Missing values before handling: {missing_before}")
        
        if strategy == 'drop':
            df = df.dropna(subset=columns)
            logger.info(f"Dropped rows with missing values. Remaining rows: {len(df)}")
        
        elif strategy == 'fill':
            if fill_value is None:
                # Use column-specific defaults
                for col in columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '', inplace=True)
            else:
                df[columns] = df[columns].fillna(fill_value)
            logger.info(f"Filled missing values with strategy: {strategy}")
        
        elif strategy == 'forward_fill':
            df[columns] = df[columns].ffill()
            df[columns] = df[columns].bfill()  # Fill remaining with backward fill
            logger.info(f"Applied forward fill strategy")
        
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'drop', 'fill', or 'forward_fill'")
        
        missing_after = df[columns].isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return df
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error handling missing values: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows with validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    subset : list, optional
        Columns to consider when identifying duplicates
    keep : str, default 'first'
        Which duplicates to keep: 'first', 'last', or False
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        df = df.copy()
        duplicates_before = df.duplicated(subset=subset).sum()
        
        if duplicates_before > 0:
            logger.info(f"Found {duplicates_before} duplicate rows")
            df = df.drop_duplicates(subset=subset, keep=keep)
            logger.info(f"Removed duplicates. Remaining rows: {len(df)}")
        else:
            logger.info("No duplicates found")
        
        return df
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error removing duplicates: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'label',
    drop_original: bool = False
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical variables with validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to encode. If None, encodes all object/category columns
    method : str, default 'label'
        Encoding method: 'label' or 'onehot'
    drop_original : bool, default False
        Whether to drop original categorical columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded columns
    Dict[str, LabelEncoder]
        Dictionary mapping column names to encoders
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        df = df.copy()
        
        if columns is None:
            # Auto-detect categorical columns
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columns:
            logger.info("No categorical columns to encode")
            return df, {}
        
        encoders = {}
        
        if method == 'label':
            for col in columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found, skipping")
                    continue
                
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder
                logger.info(f"Label encoded column: {col}")
        
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
            logger.info(f"One-hot encoded columns: {columns}")
        
        else:
            raise ValueError(f"Invalid method: {method}. Must be 'label' or 'onehot'")
        
        if drop_original and method == 'label':
            df = df.drop(columns=columns)
            logger.info(f"Dropped original columns: {columns}")
        
        return df, encoders
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error encoding categorical variables: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    scaler_type: str = 'standard',
    fit: bool = True,
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[StandardScaler, MinMaxScaler]]:
    """
    Scale numerical features with validation.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Features to scale
    scaler_type : str, default 'standard'
        Type of scaler: 'standard' or 'minmax'
    fit : bool, default True
        Whether to fit the scaler (True for training, False for testing)
    scaler : StandardScaler or MinMaxScaler, optional
        Pre-fitted scaler to use when fit=False
    
    Returns:
    --------
    pd.DataFrame or np.ndarray
        Scaled features
    StandardScaler or MinMaxScaler
        Fitted scaler
    """
    try:
        if X is None:
            raise ValueError("Input features X is None")
        
        if isinstance(X, pd.DataFrame):
            is_dataframe = True
            columns = X.columns.tolist()
            index = X.index
        else:
            is_dataframe = False
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        if X.size == 0:
            raise ValueError("Input features X is empty")
        
        # Initialize or use provided scaler
        if scaler is None:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'standard' or 'minmax'")
        
        # Fit and transform
        if fit:
            X_scaled = scaler.fit_transform(X)
            logger.info(f"Fitted and transformed features using {scaler_type} scaler")
        else:
            if not hasattr(scaler, 'mean_') and not hasattr(scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before use. Set fit=True or provide a fitted scaler.")
            X_scaled = scaler.transform(X)
            logger.info(f"Transformed features using pre-fitted {scaler_type} scaler")
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            X_scaled = pd.DataFrame(X_scaled, columns=columns, index=index)
        
        return X_scaled, scaler
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error scaling features: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def split_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray],
           Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Split data into train and test sets with validation.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    test_size : float, default 0.2
        Proportion of dataset to include in test split
    random_state : int, default 42
        Random seed for reproducibility
    stratify : pd.Series or np.ndarray, optional
        If not None, data is split in a stratified fashion using this as class labels
    
    Returns:
    --------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
        
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")
        
        logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify if stratify is not None else y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error splitting data: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_transaction_frequency_features(
    df: pd.DataFrame,
    user_id_column: str = 'user_id',
    timestamp_column: str = 'purchase_time',
    windows: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create user-level transaction frequency features for specified time windows.
    
    This function calculates the number of transactions per user within rolling
    time windows (e.g., last 1 hour, last 24 hours) to capture transaction patterns
    that may indicate fraudulent behavior.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with transaction data
    user_id_column : str, default 'user_id'
        Column name containing user identifiers
    timestamp_column : str, default 'purchase_time'
        Column name containing transaction timestamps
    windows : list, optional
        List of time windows to calculate. Options: '1h', '24h', '7d', '30d'
        If None, defaults to ['1h', '24h']
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added frequency features
    
    Raises:
    -------
    ValueError
        If required columns are missing or data is invalid
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        if user_id_column not in df.columns:
            raise ValueError(f"User ID column '{user_id_column}' not found in dataframe")
        
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in dataframe")
        
        df = df.copy()
        
        # Default windows if not specified
        if windows is None:
            windows = ['1h', '24h']
        
        # Validate windows
        valid_windows = ['1h', '24h', '7d', '30d']
        invalid_windows = [w for w in windows if w not in valid_windows]
        if invalid_windows:
            raise ValueError(f"Invalid windows: {invalid_windows}. Must be one of {valid_windows}")
        
        # Convert timestamp column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            try:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            except Exception as e:
                error_msg = f"Error converting timestamp column to datetime: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        # Sort by timestamp for rolling calculations
        df = df.sort_values(by=[user_id_column, timestamp_column]).reset_index(drop=True)
        
        logger.info(f"Creating transaction frequency features for windows: {windows}")
        
        # Create frequency features for each window
        for window in windows:
            feature_name = f'transaction_count_{window}'
            
            # Parse window duration
            if window == '1h':
                delta = timedelta(hours=1)
            elif window == '24h':
                delta = timedelta(hours=24)
            elif window == '7d':
                delta = timedelta(days=7)
            elif window == '30d':
                delta = timedelta(days=30)
            
            # Use vectorized operations for better performance
            transaction_counts = []
            
            # Group by user for efficient processing
            for user_id, user_group in df.groupby(user_id_column):
                user_timestamps = user_group[timestamp_column].values
                user_counts = []
                
                for i, current_time in enumerate(user_timestamps):
                    window_start = current_time - delta
                    # Count transactions in window (using vectorized comparison)
                    count = np.sum((user_timestamps >= window_start) & (user_timestamps <= current_time))
                    # Subtract 1 to exclude the current transaction
                    user_counts.append(count - 1)
                
                transaction_counts.extend(user_counts)
            
            df[feature_name] = transaction_counts
            logger.info(f"Created feature: {feature_name}")
        
        logger.info(f"Successfully created {len(windows)} transaction frequency features")
        return df
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error creating transaction frequency features: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_preprocessing_pipeline(
    scaler_type: str = 'standard',
    resampling_strategy: Optional[str] = None,
    random_state: int = 42
):
    """
    Create a reproducible preprocessing pipeline combining scaling and resampling.
    
    This function creates a scikit-learn compatible pipeline that can be reused
    consistently across experiments and model types. The pipeline ensures that
    scaling and resampling are applied in the correct order and can be saved/loaded
    for reproducibility.
    
    Parameters:
    -----------
    scaler_type : str, default 'standard'
        Type of scaler: 'standard' or 'minmax'
    resampling_strategy : str, optional
        Resampling strategy: 'smote', 'undersample', 'smote_undersample', or None
        If None, only scaling is applied
    random_state : int, default 42
        Random seed for reproducibility
    
    Returns:
    --------
    Pipeline
        Preprocessing pipeline that can be fitted and transformed
    
    Raises:
    -------
    ValueError
        If scaler_type or resampling_strategy is invalid
    ImportError
        If imbalanced-learn is not available and resampling is requested
    """
    try:
        if not IMBLEARN_AVAILABLE and resampling_strategy is not None:
            raise ImportError(
                "imbalanced-learn is required for resampling. "
                "Install it with: pip install imbalanced-learn"
            )
        
        # Use sklearn Pipeline if no resampling, ImbPipeline if resampling
        if resampling_strategy is None:
            from sklearn.pipeline import Pipeline
            PipelineClass = Pipeline
        else:
            PipelineClass = ImbPipeline
        steps = []
        
        # Add scaler step
        if scaler_type == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        else:
            raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'standard' or 'minmax'")
        
        # Add resampling step if specified
        if resampling_strategy is not None:
            if resampling_strategy == 'smote':
                steps.append(('resampler', SMOTE(random_state=random_state)))
            elif resampling_strategy == 'undersample':
                steps.append(('resampler', RandomUnderSampler(random_state=random_state)))
            elif resampling_strategy == 'smote_undersample':
                # Combine SMOTE and undersampling
                steps.append(('smote', SMOTE(random_state=random_state)))
                steps.append(('undersample', RandomUnderSampler(random_state=random_state)))
            else:
                raise ValueError(
                    f"Invalid resampling_strategy: {resampling_strategy}. "
                    "Must be 'smote', 'undersample', 'smote_undersample', or None"
                )
        
        pipeline = PipelineClass(steps=steps)
        logger.info(f"Created preprocessing pipeline with {len(steps)} steps")
        logger.info(f"  Steps: {[step[0] for step in steps]}")
        
        return pipeline
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error creating preprocessing pipeline: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def apply_preprocessing_pipeline(
    pipeline,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    fit: bool = True
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray],
           Optional[Union[pd.DataFrame, np.ndarray]]]:
    """
    Apply a preprocessing pipeline to training and optionally test data.
    
    This function ensures that:
    - Training data is fitted and transformed (including resampling)
    - Test data is only transformed (no resampling, using fitted scaler)
    - The pipeline can be reused consistently across experiments
    
    Parameters:
    -----------
    pipeline : Pipeline or ImbPipeline
        Preprocessing pipeline created with create_preprocessing_pipeline()
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target variable
    X_test : pd.DataFrame or np.ndarray, optional
        Test features to transform (without resampling)
    fit : bool, default True
        Whether to fit the pipeline (True for training, False if already fitted)
    
    Returns:
    --------
    Tuple of (X_train_transformed, y_train_transformed, X_test_transformed)
        X_test_transformed will be None if X_test is not provided
    """
    try:
        if pipeline is None:
            raise ValueError("Pipeline cannot be None")
        
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        # Convert to numpy arrays for consistency
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("X_train and y_train cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train must have the same length. Got {len(X_train)} and {len(y_train)}")
        
        logger.info(f"Applying preprocessing pipeline (fit={fit})")
        logger.info(f"  Training samples before: {len(X_train)}")
        
        # Fit and transform training data
        if fit:
            X_train_transformed, y_train_transformed = pipeline.fit_resample(X_train, y_train)
            logger.info(f"  Training samples after: {len(X_train_transformed)}")
        else:
            # If pipeline is already fitted, just transform
            X_train_transformed = pipeline.transform(X_train)
            y_train_transformed = y_train
            logger.info(f"  Training samples after: {len(X_train_transformed)}")
        
        # Transform test data if provided (no resampling)
        X_test_transformed = None
        if X_test is not None:
            X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            # Only apply scaling to test data (skip resampling steps)
            # Get the scaler from the pipeline
            scaler = pipeline.named_steps.get('scaler')
            if scaler is None:
                raise ValueError("Pipeline must contain a 'scaler' step")
            
            X_test_transformed = scaler.transform(X_test)
            logger.info(f"  Test samples transformed: {len(X_test_transformed)}")
        
        logger.info("Pipeline application completed successfully")
        return X_train_transformed, y_train_transformed, X_test_transformed
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error applying preprocessing pipeline: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def validate_preprocessing_input(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate input for preprocessing operations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, optional
        Name of target column to validate
    feature_columns : list, optional
        List of feature columns to validate
    
    Returns:
    --------
    bool
        True if validation passes
    
    Raises:
    -------
    ValueError
        If validation fails
    """
    try:
        if df is None or df.empty:
            raise ValueError("Dataframe is None or empty")
        
        if target_column is not None:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        if feature_columns is not None:
            missing_columns = set(feature_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Feature columns not found: {missing_columns}")
        
        logger.info("Preprocessing input validation passed")
        return True
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Preprocessing validation error: {str(e)}"
        logger.error(error_msg)
        raise

