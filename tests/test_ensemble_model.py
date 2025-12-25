"""
Unit tests for EnsembleModel class.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.ensemble_model import EnsembleModel, EnsembleModelResults


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        weights=[0.9, 0.1]
    )
    return X, y


@pytest.fixture
def split_data(sample_data):
    """Split data into train and test sets."""
    X, y = sample_data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


class TestEnsembleModel:
    """Test suite for EnsembleModel class."""
    
    def test_init_random_forest(self):
        """Test Random Forest initialization."""
        model = EnsembleModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert model.class_weight == 'balanced'
        assert model.model is None
    
    def test_init_xgboost(self):
        """Test XGBoost initialization."""
        model = EnsembleModel(model_type='xgboost')
        assert model.model_type == 'xgboost'
    
    def test_init_lightgbm(self):
        """Test LightGBM initialization."""
        model = EnsembleModel(model_type='lightgbm')
        assert model.model_type == 'lightgbm'
    
    def test_init_invalid_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            EnsembleModel(model_type='invalid')
    
    def test_train_random_forest(self, split_data):
        """Test Random Forest training."""
        X_train, _, y_train, _ = split_data
        model = EnsembleModel(model_type='random_forest', random_state=42)
        trained_model = model.train(X_train, y_train, n_estimators=10, max_depth=5)
        
        assert model.model is not None
        assert hasattr(trained_model, 'predict')
    
    def test_train_xgboost(self, split_data):
        """Test XGBoost training."""
        X_train, _, y_train, _ = split_data
        model = EnsembleModel(model_type='xgboost', random_state=42)
        trained_model = model.train(X_train, y_train, n_estimators=10, max_depth=3)
        
        assert model.model is not None
        assert hasattr(trained_model, 'predict')
    
    def test_tune_hyperparameters(self, split_data):
        """Test hyperparameter tuning."""
        X_train, _, y_train, _ = split_data
        model = EnsembleModel(model_type='random_forest', random_state=42)
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        results = model.tune_hyperparameters(
            X_train, y_train,
            param_grid=param_grid,
            cv=3,
            search_type='grid'
        )
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert model.model is not None
        assert model.best_params is not None
    
    def test_evaluate(self, split_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = split_data
        model = EnsembleModel(model_type='random_forest', random_state=42)
        model.train(X_train, y_train, n_estimators=10)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'f1' in metrics
        assert 'pr_auc' in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if v is not None)
    
    def test_train_and_evaluate(self, split_data):
        """Test complete train and evaluate pipeline."""
        X_train, X_test, y_train, y_test = split_data
        model = EnsembleModel(model_type='random_forest', random_state=42)
        
        results = model.train_and_evaluate(
            X_train, y_train, X_test, y_test,
            param_grid={'n_estimators': [10], 'max_depth': [5]},
            tune_hyperparameters=False
        )
        
        assert isinstance(results, EnsembleModelResults)
        assert results.model is not None
        assert 'f1' in results.test_metrics
        assert results.confusion_matrix.shape == (2, 2)
    
    def test_get_feature_importance(self, split_data):
        """Test feature importance extraction."""
        X_train, _, y_train, _ = split_data
        model = EnsembleModel(model_type='random_forest', random_state=42)
        model.train(X_train, y_train, n_estimators=10)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert 'importance' in importance.columns

