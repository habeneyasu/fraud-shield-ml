"""
Unit tests for ModelComparator class.
"""

import pytest
import numpy as np
from src.model_comparison import (
    ModelComparator, ModelComparisonEntry, ModelComparisonResults
)


@pytest.fixture
def sample_model_entries():
    """Create sample model comparison entries."""
    entries = [
        ModelComparisonEntry(
            model_name='Model A',
            model_type='baseline',
            test_metrics={'f1': 0.85, 'pr_auc': 0.82, 'roc_auc': 0.90},
            interpretability_score=1.0
        ),
        ModelComparisonEntry(
            model_name='Model B',
            model_type='random_forest',
            test_metrics={'f1': 0.90, 'pr_auc': 0.88, 'roc_auc': 0.92},
            interpretability_score=0.6
        ),
        ModelComparisonEntry(
            model_name='Model C',
            model_type='xgboost',
            test_metrics={'f1': 0.88, 'pr_auc': 0.85, 'roc_auc': 0.91},
            interpretability_score=0.5
        )
    ]
    return entries


class TestModelComparator:
    """Test suite for ModelComparator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        comparator = ModelComparator()
        assert comparator.primary_metric == 'f1'
        assert comparator.interpretability_weight == 0.3
        assert comparator.performance_weight == 0.7
    
    def test_init_custom(self):
        """Test custom initialization."""
        comparator = ModelComparator(
            primary_metric='pr_auc',
            interpretability_weight=0.5,
            performance_weight=0.5
        )
        assert comparator.primary_metric == 'pr_auc'
        assert comparator.interpretability_weight == 0.5
        assert comparator.performance_weight == 0.5
    
    def test_init_invalid_metric(self):
        """Test initialization with invalid primary metric."""
        with pytest.raises(ValueError, match="Invalid primary_metric"):
            ModelComparator(primary_metric='invalid')
    
    def test_init_invalid_weights(self):
        """Test initialization with weights that don't sum to 1.0."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            ModelComparator(interpretability_weight=0.3, performance_weight=0.5)
    
    def test_compare_models(self, sample_model_entries):
        """Test model comparison."""
        comparator = ModelComparator(primary_metric='f1')
        results = comparator.compare_models(sample_model_entries)
        
        assert isinstance(results, ModelComparisonResults)
        assert results.best_model_name in ['Model A', 'Model B', 'Model C']
        assert len(results.ranking) == 3
        assert len(results.comparison_df) == 3
    
    def test_compare_models_empty(self):
        """Test comparison with empty list."""
        comparator = ModelComparator()
        with pytest.raises(ValueError, match="cannot be empty"):
            comparator.compare_models([])
    
    def test_comparison_df_structure(self, sample_model_entries):
        """Test comparison DataFrame structure."""
        comparator = ModelComparator()
        results = comparator.compare_models(sample_model_entries)
        
        df = results.comparison_df
        assert 'model_name' in df.columns
        assert 'model_type' in df.columns
        assert 'interpretability_score' in df.columns
        assert 'performance_score' in df.columns
        assert 'overall_score' in df.columns
    
    def test_ranking(self, sample_model_entries):
        """Test model ranking."""
        comparator = ModelComparator()
        results = comparator.compare_models(sample_model_entries)
        
        assert len(results.ranking) == 3
        assert results.best_model_name == results.ranking[0]
        assert all(model in results.ranking for model in ['Model A', 'Model B', 'Model C'])
    
    def test_justification(self, sample_model_entries):
        """Test justification generation."""
        comparator = ModelComparator()
        results = comparator.compare_models(sample_model_entries)
        
        assert len(results.best_model_justification) > 0
        assert results.best_model_name in results.best_model_justification
    
    def test_get_comparison_summary(self, sample_model_entries):
        """Test comparison summary generation."""
        comparator = ModelComparator()
        results = comparator.compare_models(sample_model_entries)
        summary = comparator.get_comparison_summary(results)
        
        assert 'best_model' in summary
        assert 'best_overall_score' in summary
        assert 'justification' in summary
        assert 'ranking' in summary

