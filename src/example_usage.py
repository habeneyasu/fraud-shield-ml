"""
Example Usage of Reusable Modules

This script demonstrates how to use the data loading, preprocessing, and modeling
modules with proper error handling.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, save_dataframe, validate_dataframe
from src.preprocessing import (
    handle_missing_values,
    remove_duplicates,
    encode_categorical,
    create_transaction_frequency_features,
    create_preprocessing_pipeline,
    apply_preprocessing_pipeline,
    split_data
)
from src.modeling import train_model, evaluate_model, save_model, load_model


def main():
    """Example workflow demonstrating module usage."""
    
    print("=" * 80)
    print("Example: Using Reusable Modules for Fraud Detection")
    print("=" * 80)
    
    # 1. Data Loading with error handling
    print("\n1. Loading Data...")
    try:
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
        print(f"✓ Successfully loaded {len(df)} rows")
        
        # Validate the dataframe
        validate_dataframe(df, min_rows=1, required_columns=['class'])
        print("✓ Dataframe validation passed")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    except ValueError as e:
        print(f"✗ Validation error: {e}")
        return
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return
    
    # 2. Preprocessing with error handling
    print("\n2. Preprocessing Data...")
    try:
        # Handle missing values
        df_clean = handle_missing_values(df, strategy='drop')
        print(f"✓ Handled missing values. Remaining rows: {len(df_clean)}")
        
        # Remove duplicates
        df_clean = remove_duplicates(df_clean)
        print(f"✓ Removed duplicates. Remaining rows: {len(df_clean)}")
        
        # Create transaction frequency features (NEW FEATURE)
        print("\n2a. Creating Transaction Frequency Features...")
        df_clean = create_transaction_frequency_features(
            df_clean,
            user_id_column='user_id',
            timestamp_column='purchase_time',
            windows=['1h', '24h']  # Last 1 hour and 24 hours
        )
        print("✓ Created transaction frequency features (transaction_count_1h, transaction_count_24h)")
        
        # Encode categorical variables
        categorical_cols = ['source', 'browser', 'sex', 'device_id']
        df_encoded, encoders = encode_categorical(
            df_clean,
            columns=categorical_cols,
            method='label'
        )
        print(f"✓ Encoded {len(categorical_cols)} categorical columns")
        
    except ValueError as e:
        print(f"✗ Preprocessing error: {e}")
        return
    except Exception as e:
        print(f"✗ Unexpected preprocessing error: {e}")
        return
    
    # 3. Prepare features and target
    print("\n3. Preparing Features and Target...")
    try:
        # Select features (excluding target and non-feature columns)
        # Note: transaction_count_1h and transaction_count_24h are now included
        feature_cols = [col for col in df_encoded.columns 
                       if col not in ['class', 'user_id', 'signup_time', 'purchase_time', 'ip_address']]
        X = df_encoded[feature_cols]
        y = df_encoded['class']
        
        print(f"✓ Features: {len(feature_cols)} columns")
        print(f"  Includes transaction frequency features: transaction_count_1h, transaction_count_24h")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"✗ Error preparing features: {e}")
        return
    
    # 4. Split data (before scaling/resampling pipeline)
    print("\n4. Splitting Data...")
    try:
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=0.2,
            random_state=42
        )
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
    except Exception as e:
        print(f"✗ Error splitting data: {e}")
        return
    
    # 5. Create and apply reproducible preprocessing pipeline (NEW FEATURE)
    print("\n5. Creating Preprocessing Pipeline...")
    try:
        # Create a reproducible pipeline with scaling and resampling
        pipeline = create_preprocessing_pipeline(
            scaler_type='standard',
            resampling_strategy='smote',  # Options: 'smote', 'undersample', 'smote_undersample', None
            random_state=42
        )
        print("✓ Preprocessing pipeline created")
        print("  Pipeline steps: scaling + resampling")
        
        # Apply pipeline to training and test data
        X_train_processed, y_train_processed, X_test_processed = apply_preprocessing_pipeline(
            pipeline,
            X_train, y_train,
            X_test=X_test,
            fit=True
        )
        print(f"✓ Pipeline applied successfully")
        print(f"  Training samples after resampling: {len(X_train_processed)}")
        print(f"  Test samples (scaled only): {len(X_test_processed)}")
        
        # Update variables for model training
        X_train = X_train_processed
        y_train = y_train_processed
        X_test = X_test_processed
        
    except Exception as e:
        print(f"✗ Error applying preprocessing pipeline: {e}")
        return
    
    # 6. Train model with error handling
    print("\n6. Training Model...")
    try:
        model = train_model(
            X_train, y_train,
            model_type='random_forest',
            model_params={'n_estimators': 100, 'max_depth': 10},
            class_weight='balanced'
        )
        print("✓ Model trained successfully")
        
    except ValueError as e:
        print(f"✗ Training error: {e}")
        return
    except Exception as e:
        print(f"✗ Unexpected training error: {e}")
        return
    
    # 7. Evaluate model
    print("\n7. Evaluating Model...")
    try:
        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test)
        print("✓ Test Set Metrics:")
        for metric, value in test_metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        # Evaluate on train set
        train_metrics = evaluate_model(model, X_train, y_train)
        print("\n✓ Train Set Metrics:")
        for metric, value in train_metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return
    
    # 8. Save model
    print("\n8. Saving Model...")
    try:
        model_path = Path('../models') / 'example_fraud_model.joblib'
        save_model(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return
    
    # 9. Load model (demonstration)
    print("\n9. Loading Model...")
    try:
        loaded_model = load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Verify loaded model works
        test_pred = loaded_model.predict(X_test[:5])
        print(f"✓ Loaded model predictions: {test_pred}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

