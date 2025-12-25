"""
Fraud Shield ML - Source Package

This package provides reusable modules for data loading, preprocessing, and modeling
with proper error handling and validation checks.
"""

from . import data_loader
from . import preprocessing
from . import modeling
from . import analysis
from . import data_preparation
from . import baseline_model
from . import ensemble_model
from . import cross_validation
from . import model_comparison

# Import commonly used functions for convenience
from .data_loader import (
    load_csv,
    load_fraud_data,
    load_creditcard_data,
    load_ip_mapping,
    save_dataframe,
    validate_dataframe
)

from .preprocessing import (
    handle_missing_values,
    remove_duplicates,
    encode_categorical,
    scale_features,
    split_data,
    validate_preprocessing_input,
    create_transaction_frequency_features,
    create_preprocessing_pipeline,
    apply_preprocessing_pipeline
)

from .modeling import (
    train_model,
    evaluate_model,
    get_classification_report,
    get_confusion_matrix,
    save_model,
    load_model,
    cross_validate_model,
    predict
)

from .analysis import (
    analyze_amount_vs_fraud,
    analyze_device_vs_fraud,
    analyze_source_vs_fraud,
    analyze_browser_vs_fraud,
    generate_risk_summary_report
)

from .data_preparation import (
    DataPreparation,
    DataSplitResult
)

from .baseline_model import (
    BaselineModel,
    BaselineModelResults
)

from .ensemble_model import (
    EnsembleModel,
    EnsembleModelResults
)

from .cross_validation import (
    CrossValidator,
    CrossValidationResults
)

from .model_comparison import (
    ModelComparator,
    ModelComparisonEntry,
    ModelComparisonResults
)

__all__ = [
    # Modules
    'data_loader',
    'preprocessing',
    'modeling',
    'analysis',
    'data_preparation',
    'baseline_model',
    'ensemble_model',
    'cross_validation',
    'model_comparison',
    
    # Data loading functions
    'load_csv',
    'load_fraud_data',
    'load_creditcard_data',
    'load_ip_mapping',
    'save_dataframe',
    'validate_dataframe',
    
    # Preprocessing functions
    'handle_missing_values',
    'remove_duplicates',
    'encode_categorical',
    'scale_features',
    'split_data',
    'validate_preprocessing_input',
    'create_transaction_frequency_features',
    'create_preprocessing_pipeline',
    'apply_preprocessing_pipeline',
    
    # Modeling functions
    'train_model',
    'evaluate_model',
    'get_classification_report',
    'get_confusion_matrix',
    'save_model',
    'load_model',
    'cross_validate_model',
    'predict',
    
    # Analysis functions
    'analyze_amount_vs_fraud',
    'analyze_device_vs_fraud',
    'analyze_source_vs_fraud',
    'analyze_browser_vs_fraud',
    'generate_risk_summary_report',
    
    # Data preparation classes
    'DataPreparation',
    'DataSplitResult',
    
    # Baseline model classes
    'BaselineModel',
    'BaselineModelResults',
    
    # Ensemble model classes
    'EnsembleModel',
    'EnsembleModelResults',
    
    # Cross-validation classes
    'CrossValidator',
    'CrossValidationResults',
    
    # Model comparison classes
    'ModelComparator',
    'ModelComparisonEntry',
    'ModelComparisonResults',
]

__version__ = '1.0.0'

