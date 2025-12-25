"""
Unit tests for BaselineModel class.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.baseline_model import BaselineModel, BaselineModelResults


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


class TestBaselineModel:
    """Test suite for BaselineModel class."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = BaselineModel()
        assert model.class_weight == 'balanced'
        assert model.random_state == 42
        assert model.max_iter == 1000
        assert model.model is None
    
    def test_init_custom(self):
        """Test custom initialization."""
        model = BaselineModel(
            class_weight=None,
            random_state=123,
            max_iter=500,
            solver='liblinear'
        )
        assert model.class_weight is None
        assert model.random_state == 123
        assert model.max_iter == 500
        assert model.solver == 'liblinear'
    
    def test_init_invalid_class_weight(self):
        """Test initialization with invalid class_weight."""
        with pytest.raises(ValueError, match="Invalid class_weight"):
            BaselineModel(class_weight='invalid')
    
    def test_init_invalid_solver(self):
        """Test initialization with invalid solver."""
        with pytest.raises(ValueError, match="Invalid solver"):
            BaselineModel(solver='invalid')
    
    def test_train(self, split_data):
        """Test model training."""
        X_train, _, y_train, _ = split_data
        model = BaselineModel(random_state=42)
        trained_model = model.train(X_train, y_train)
        
        assert model.model is not None
        assert hasattr(trained_model, 'predict')
        assert hasattr(trained_model, 'predict_proba')
    
    def test_train_invalid_input(self):
        """Test training with invalid input."""
        model = BaselineModel()
        with pytest.raises(ValueError):
            model.train(None, None)
    
    def test_evaluate(self, split_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = split_data
        model = BaselineModel(random_state=42)
        model.train(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'pr_auc' in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if v is not None)
    
    def test_evaluate_not_trained(self, split_data):
        """Test evaluation before training."""
        _, X_test, _, y_test = split_data
        model = BaselineModel()
        with pytest.raises(ValueError, match="Model has not been trained"):
            model.evaluate(X_test, y_test)
    
    def test_get_confusion_matrix(self, split_data):
        """Test confusion matrix generation."""
        X_train, X_test, y_train, y_test = split_data
        model = BaselineModel(random_state=42)
        model.train(X_train, y_train)
        
        cm = model.get_confusion_matrix(X_test, y_test)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_test)
    
    def test_get_classification_report(self, split_data):
        """Test classification report generation."""
        X_train, X_test, y_train, y_test = split_data
        model = BaselineModel(random_state=42)
        model.train(X_train, y_train)
        
        report = model.get_classification_report(X_test, y_test)
        
        assert isinstance(report, str)
        assert 'precision' in report.lower()
        assert 'recall' in report.lower()
    
    def test_train_and_evaluate(self, split_data):
        """Test complete train and evaluate pipeline."""
        X_train, X_test, y_train, y_test = split_data
        model = BaselineModel(random_state=42)
        results = model.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        assert isinstance(results, BaselineModelResults)
        assert results.model is not None
        assert 'f1' in results.test_metrics
        assert 'pr_auc' in results.test_metrics
        assert results.confusion_matrix.shape == (2, 2)
    
    def test_get_feature_importance(self, split_data):
        """Test feature importance extraction."""
        X_train, _, y_train, _ = split_data
        model = BaselineModel(random_state=42)
        model.train(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert 'coefficient' in importance.columns
        assert 'abs_coefficient' in importance.columns

