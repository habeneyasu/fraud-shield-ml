"""
Data Loading Module

This module provides reusable functions for loading data files with proper
error handling, validation checks, and I/O operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv(
    file_path: Union[str, Path],
    required_columns: Optional[List[str]] = None,
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
    """
    Load a CSV file with error handling and validation.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file to load
    required_columns : list, optional
        List of column names that must be present in the dataset
    encoding : str, default 'utf-8'
        Encoding to use when reading the file
    **kwargs
        Additional arguments to pass to pd.read_csv()
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    
    Raises:
    -------
    FileNotFoundError
        If the file does not exist
    ValueError
        If required columns are missing or data is invalid
    pd.errors.EmptyDataError
        If the file is empty
    """
    file_path = Path(file_path)
    
    try:
        # Check if file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check if file is readable
        if not file_path.is_file():
            error_msg = f"Path is not a file: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Loading CSV file: {file_path}")
        
        # Attempt to load the CSV file
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        except pd.errors.EmptyDataError as e:
            error_msg = f"File is empty: {file_path}"
            logger.error(error_msg)
            raise pd.errors.EmptyDataError(error_msg) from e
        except UnicodeDecodeError as e:
            error_msg = f"Encoding error when reading {file_path}. Try a different encoding."
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Error reading CSV file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise
        
        # Validate that dataframe is not empty
        if df.empty:
            error_msg = f"Loaded dataframe is empty: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Validate required columns if specified
        if required_columns is not None:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"All required columns present: {required_columns}")
        
        return df
        
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error loading file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_fraud_data(
    data_dir: Union[str, Path] = '../data',
    filename: str = 'Fraud_Data.csv',
    subdirectory: str = 'raw'
) -> pd.DataFrame:
    """
    Load the fraud e-commerce dataset with validation.
    
    Parameters:
    -----------
    data_dir : str or Path, default '../data'
        Base data directory
    filename : str, default 'Fraud_Data.csv'
        Name of the fraud data file
    subdirectory : str, default 'raw'
        Subdirectory within data_dir containing the file
    
    Returns:
    --------
    pd.DataFrame
        Loaded fraud dataset
    """
    data_dir = Path(data_dir)
    file_path = data_dir / subdirectory / filename
    
    required_columns = [
        'user_id', 'signup_time', 'purchase_time', 'purchase_value',
        'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'
    ]
    
    return load_csv(file_path, required_columns=required_columns)


def load_creditcard_data(
    data_dir: Union[str, Path] = '../data',
    filename: str = 'creditcard.csv',
    subdirectory: str = 'raw'
) -> pd.DataFrame:
    """
    Load the credit card fraud dataset with validation.
    
    Parameters:
    -----------
    data_dir : str or Path, default '../data'
        Base data directory
    filename : str, default 'creditcard.csv'
        Name of the credit card data file
    subdirectory : str, default 'raw'
        Subdirectory within data_dir containing the file
    
    Returns:
    --------
    pd.DataFrame
        Loaded credit card dataset
    """
    data_dir = Path(data_dir)
    file_path = data_dir / subdirectory / filename
    
    required_columns = ['Class']  # At minimum, Class column must be present
    
    return load_csv(file_path, required_columns=required_columns)


def load_ip_mapping(
    data_dir: Union[str, Path] = '../data',
    filename: str = 'IpAddress_to_Country.csv',
    subdirectory: str = 'raw'
) -> pd.DataFrame:
    """
    Load the IP address to country mapping file with validation.
    
    Parameters:
    -----------
    data_dir : str or Path, default '../data'
        Base data directory
    filename : str, default 'IpAddress_to_Country.csv'
        Name of the IP mapping file
    subdirectory : str, default 'raw'
        Subdirectory within data_dir containing the file
    
    Returns:
    --------
    pd.DataFrame
        Loaded IP mapping dataset
    """
    data_dir = Path(data_dir)
    file_path = data_dir / subdirectory / filename
    
    return load_csv(file_path)


def save_dataframe(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    create_dirs: bool = True,
    **kwargs
) -> None:
    """
    Save a dataframe to CSV with error handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    file_path : str or Path
        Path where to save the file
    create_dirs : bool, default True
        Whether to create parent directories if they don't exist
    **kwargs
        Additional arguments to pass to df.to_csv()
    
    Raises:
    -------
    ValueError
        If dataframe is empty or invalid
    PermissionError
        If file cannot be written due to permissions
    """
    file_path = Path(file_path)
    
    try:
        # Validate dataframe
        if df is None:
            error_msg = "Dataframe is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if df.empty:
            error_msg = "Cannot save empty dataframe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataframe to: {file_path}")
        logger.info(f"Shape: {df.shape}")
        
        # Attempt to save the file
        try:
            df.to_csv(file_path, index=False, **kwargs)
        except PermissionError as e:
            error_msg = f"Permission denied when writing to {file_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg) from e
        except Exception as e:
            error_msg = f"Error saving dataframe to {file_path}: {str(e)}"
            logger.error(error_msg)
            raise
        
        # Verify file was created
        if not file_path.exists():
            error_msg = f"File was not created: {file_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"Successfully saved dataframe to: {file_path}")
        
    except (ValueError, PermissionError, RuntimeError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error saving file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 1,
    required_columns: Optional[List[str]] = None,
    check_duplicates: bool = False
) -> bool:
    """
    Validate a dataframe meets basic requirements.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    min_rows : int, default 1
        Minimum number of rows required
    required_columns : list, optional
        List of column names that must be present
    check_duplicates : bool, default False
        Whether to check for duplicate rows
    
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
        if df is None:
            raise ValueError("Dataframe is None")
        
        if df.empty:
            raise ValueError("Dataframe is empty")
        
        if len(df) < min_rows:
            raise ValueError(f"Dataframe has {len(df)} rows, minimum required: {min_rows}")
        
        if required_columns is not None:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        if check_duplicates:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate rows in dataframe")
        
        logger.info("Dataframe validation passed")
        return True
        
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error during dataframe validation: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

