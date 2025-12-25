"""
Example: Using DataPreparation Class

This script demonstrates how to use the DataPreparation class for preparing
data for model training with both e-commerce and banking datasets.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, load_creditcard_data
from src.data_preparation import DataPreparation, DataSplitResult


def example_ecommerce_dataset():
    """Example using e-commerce fraud dataset (Fraud_Data.csv)."""
    print("=" * 80)
    print("Example 1: E-commerce Dataset (Fraud_Data.csv)")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    try:
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Initialize DataPreparation for e-commerce dataset
    print("\n2. Initializing DataPreparation...")
    prep = DataPreparation(
        dataset_type='ecommerce',
        test_size=0.2,
        random_state=42,
        stratify=True
    )
    
    # Get feature information
    print("\n3. Feature information:")
    feature_info = prep.get_feature_info(df)
    print(f"  Total columns: {feature_info['total_columns']}")
    print(f"  Target column: {feature_info['target_column']}")
    print(f"  Number of features: {feature_info['num_features']}")
    print(f"  Excluded columns: {feature_info['excluded_columns']}")
    
    # Prepare and split data
    print("\n4. Preparing and splitting data...")
    try:
        result: DataSplitResult = prep.prepare_and_split(df)
        
        print(f"\n✓ Data preparation completed:")
        print(f"  Training set: {result.train_size} samples")
        print(f"  Test set: {result.test_size} samples")
        print(f"  Training class distribution: {result.train_class_distribution}")
        print(f"  Test class distribution: {result.test_class_distribution}")
        
        # Access the split data
        X_train = result.X_train
        X_test = result.X_test
        y_train = result.y_train
        y_test = result.y_test
        
        print(f"\n  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Example 1 completed successfully!")


def example_banking_dataset():
    """Example using banking credit card fraud dataset (creditcard.csv)."""
    print("\n" + "=" * 80)
    print("Example 2: Banking Dataset (creditcard.csv)")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    try:
        df = load_creditcard_data(data_dir='../data', filename='creditcard.csv')
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Initialize DataPreparation for banking dataset
    print("\n2. Initializing DataPreparation...")
    prep = DataPreparation(
        dataset_type='banking',
        test_size=0.2,
        random_state=42,
        stratify=True
    )
    
    # Get feature information
    print("\n3. Feature information:")
    feature_info = prep.get_feature_info(df)
    print(f"  Total columns: {feature_info['total_columns']}")
    print(f"  Target column: {feature_info['target_column']}")
    print(f"  Number of features: {feature_info['num_features']}")
    print(f"  Excluded columns: {feature_info['excluded_columns']}")
    
    # Prepare and split data
    print("\n4. Preparing and splitting data...")
    try:
        result: DataSplitResult = prep.prepare_and_split(df)
        
        print(f"\n✓ Data preparation completed:")
        print(f"  Training set: {result.train_size} samples")
        print(f"  Test set: {result.test_size} samples")
        print(f"  Training class distribution: {result.train_class_distribution}")
        print(f"  Test class distribution: {result.test_class_distribution}")
        
        # Access the split data
        X_train = result.X_train
        X_test = result.X_test
        y_train = result.y_train
        y_test = result.y_test
        
        print(f"\n  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Example 2 completed successfully!")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Configuration")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    try:
        df = load_fraud_data(data_dir='../data', filename='Fraud_Data.csv')
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Custom configuration: exclude additional columns
    print("\n2. Initializing with custom configuration...")
    prep = DataPreparation(
        dataset_type='ecommerce',
        exclude_columns=['user_id', 'signup_time', 'purchase_time', 'ip_address', 'device_id'],
        test_size=0.25,  # 75/25 split instead of 80/20
        random_state=123,
        stratify=True
    )
    
    # Use specific feature columns
    print("\n3. Using specific feature columns...")
    feature_columns = ['purchase_value', 'age', 'sex', 'source', 'browser']
    
    try:
        result = prep.prepare_and_split(df, feature_columns=feature_columns)
        
        print(f"\n✓ Custom data preparation completed:")
        print(f"  Features used: {feature_columns}")
        print(f"  Training set: {result.train_size} samples")
        print(f"  Test set: {result.test_size} samples")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Example 3 completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DataPreparation Class - Usage Examples")
    print("=" * 80)
    
    # Run examples
    example_ecommerce_dataset()
    example_banking_dataset()
    example_custom_configuration()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

