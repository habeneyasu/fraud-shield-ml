"""
Unit tests for CrossValidator class.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.cross_validation import CrossValidator, CrossValidationResults
from src.baseline_model import BaselineModel


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


class TestCrossValidator:
    """Test suite for CrossValidator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        cv = CrossValidator()
        assert cv.n_folds == 5
        assert cv.random_state == 42
        assert 'f1' in cv.metrics
        assert 'pr_auc' in cv.metrics
    
    def test_init_custom(self):
        """Test custom initialization."""
        cv = CrossValidator(
            n_folds=10,
            random_state=123,
            metrics=['f1', 'roc_auc']
        )
        assert cv.n_folds == 10
        assert cv.random_state == 123
        assert len(cv.metrics) == 2
    
    def test_init_invalid_folds(self):
        """Test initialization with invalid n_folds."""
        with pytest.raises(ValueError, match="n_folds must be at least 2"):
            CrossValidator(n_folds=1)
    
    def test_init_invalid_metrics(self):
        """Test initialization with invalid metrics."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            CrossValidator(metrics=['invalid_metric'])
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        model = BaselineModel(random_state=42)
        model.train(X, y)
        
        cv = CrossValidator(n_folds=5, random_state=42)
        results = cv.cross_validate(model, X, y)
        
        assert isinstance(results, CrossValidationResults)
        assert results.n_folds == 5
        assert 'f1' in results.metrics_summary
        assert 'pr_auc' in results.metrics_summary
        assert len(results.fold_results) == 5
    
    def test_cross_validate_metrics_summary(self, sample_data):
        """Test metrics summary structure."""
        X, y = sample_data
        model = BaselineModel(random_state=42)
        model.train(X, y)
        
        cv = CrossValidator(n_folds=3, random_state=42)
        results = cv.cross_validate(model, X, y)
        
        for metric, stats in results.metrics_summary.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
    
    def test_cross_validate_invalid_input(self):
        """Test cross-validation with invalid input."""
        cv = CrossValidator()
        with pytest.raises(ValueError):
            cv.cross_validate(None, None, None)

