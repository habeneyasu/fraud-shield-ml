"""
Unit tests for DataPreparation class.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preparation import DataPreparation, DataSplitResult


@pytest.fixture
def sample_ecommerce_data():
    """Create sample e-commerce fraud data."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'user_id': [f'user_{i}' for i in range(n_samples)],
        'purchase_value': np.random.uniform(10, 100, n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_banking_data():
    """Create sample banking fraud data."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'V1': np.random.randn(n_samples),
        'V2': np.random.randn(n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    return pd.DataFrame(data)


class TestDataPreparation:
    """Test suite for DataPreparation class."""
    
    def test_init_ecommerce(self):
        """Test initialization for e-commerce dataset."""
        prep = DataPreparation(dataset_type='ecommerce')
        assert prep.dataset_type == 'ecommerce'
        assert prep.target_column == 'class'
        assert 'user_id' in prep.exclude_columns
    
    def test_init_banking(self):
        """Test initialization for banking dataset."""
        prep = DataPreparation(dataset_type='banking')
        assert prep.dataset_type == 'banking'
        assert prep.target_column == 'Class'
        assert 'Time' in prep.exclude_columns
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        prep = DataPreparation(
            dataset_type='ecommerce',
            exclude_columns=['custom_col'],
            test_size=0.25,
            random_state=123
        )
        assert prep.exclude_columns == ['custom_col']
        assert prep.test_size == 0.25
        assert prep.random_state == 123
    
    def test_init_invalid_dataset_type(self):
        """Test initialization with invalid dataset type."""
        with pytest.raises(ValueError, match="Invalid dataset_type"):
            DataPreparation(dataset_type='invalid')
    
    def test_init_invalid_test_size(self):
        """Test initialization with invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be between"):
            DataPreparation(test_size=1.5)
    
    def test_validate_dataframe(self, sample_ecommerce_data):
        """Test dataframe validation."""
        prep = DataPreparation(dataset_type='ecommerce')
        assert prep.validate_dataframe(sample_ecommerce_data) is True
    
    def test_validate_dataframe_missing_target(self, sample_ecommerce_data):
        """Test validation fails when target column is missing."""
        prep = DataPreparation(dataset_type='ecommerce')
        df_no_target = sample_ecommerce_data.drop(columns=['class'])
        with pytest.raises(ValueError, match="Target column"):
            prep.validate_dataframe(df_no_target)
    
    def test_separate_features_target(self, sample_ecommerce_data):
        """Test feature-target separation."""
        prep = DataPreparation(dataset_type='ecommerce')
        X, y = prep.separate_features_target(sample_ecommerce_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'class' not in X.columns
        assert 'user_id' not in X.columns
        assert len(X) == len(y)
        assert len(X.columns) > 0
    
    def test_split_data(self, sample_ecommerce_data):
        """Test data splitting."""
        prep = DataPreparation(dataset_type='ecommerce', test_size=0.2)
        X, y = prep.separate_features_target(sample_ecommerce_data)
        result = prep.split_data(X, y)
        
        assert isinstance(result, DataSplitResult)
        assert len(result.X_train) + len(result.X_test) == len(X)
        assert len(result.y_train) + len(result.y_test) == len(y)
        assert result.train_size > 0
        assert result.test_size > 0
        assert len(result.train_class_distribution) > 0
    
    def test_prepare_and_split(self, sample_ecommerce_data):
        """Test complete prepare and split pipeline."""
        prep = DataPreparation(dataset_type='ecommerce', test_size=0.2)
        result = prep.prepare_and_split(sample_ecommerce_data)
        
        assert isinstance(result, DataSplitResult)
        assert result.train_size > 0
        assert result.test_size > 0
        assert len(result.X_train.columns) > 0
    
    def test_get_feature_info(self, sample_ecommerce_data):
        """Test feature information retrieval."""
        prep = DataPreparation(dataset_type='ecommerce')
        info = prep.get_feature_info(sample_ecommerce_data)
        
        assert 'num_features' in info
        assert 'feature_columns' in info
        assert info['num_features'] > 0

