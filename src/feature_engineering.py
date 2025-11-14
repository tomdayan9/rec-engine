"""
Feature engineering module for the Insurance Recommendation Engine.
Handles feature extraction, encoding, and scaling.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Tuple, List, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for insurance recommendation models."""

    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.one_hot_columns = []

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw demographic data.

        Args:
            df: DataFrame with demographic features

        Returns:
            DataFrame with additional engineered features
        """
        df_features = df.copy()

        # Age group
        df_features['age_group'] = pd.cut(
            df_features['Age as of Billing Start Date'],
            bins=config.AGE_BRACKETS,
            labels=config.AGE_LABELS,
            include_lowest=True
        )

        # Income bracket
        df_features['income_bracket'] = pd.cut(
            df_features['Estimated Annual Gross Income'],
            bins=config.INCOME_BRACKETS,
            labels=config.INCOME_LABELS,
            include_lowest=True
        )

        return df_features

    def encode_categorical_features(self, df: pd.DataFrame,
                                   is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.

        Args:
            df: DataFrame with categorical features
            is_training: Whether this is training data (fit encoders) or test data (transform only)

        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()

        # Categorical columns to encode
        categorical_cols = ['Sex', 'State', 'Employment Status', 'Job Class',
                           'age_group', 'income_bracket']

        if is_training:
            # One-hot encode and store column names
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols,
                                       prefix=categorical_cols, drop_first=False)
            self.one_hot_columns = [col for col in df_encoded.columns
                                   if any(cat in col for cat in categorical_cols)]
            logger.info(f"Created {len(self.one_hot_columns)} one-hot encoded features")
        else:
            # Use stored column names from training
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols,
                                       prefix=categorical_cols, drop_first=False)

            # Ensure all training columns exist
            for col in self.one_hot_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            # Remove any extra columns not in training
            extra_cols = [col for col in df_encoded.columns
                         if col not in self.one_hot_columns and
                         any(cat in col for cat in categorical_cols)]
            df_encoded = df_encoded.drop(columns=extra_cols)

        return df_encoded

    def scale_numeric_features(self, df: pd.DataFrame,
                              is_training: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.

        Args:
            df: DataFrame with numeric features
            is_training: Whether this is training data (fit scaler) or test data (transform only)

        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()

        # Numeric columns to scale
        numeric_cols = ['Age as of Billing Start Date', 'Pay Frequency',
                       'Estimated Annual Gross Income']

        if is_training:
            # Fit and transform
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            self.scalers['numeric'] = scaler
            logger.info(f"Fitted scaler for {len(numeric_cols)} numeric features")
        else:
            # Transform only
            if 'numeric' in self.scalers:
                df_scaled[numeric_cols] = self.scalers['numeric'].transform(df_scaled[numeric_cols])
            else:
                logger.warning("No fitted scaler found. Using unscaled features.")

        return df_scaled

    def prepare_features(self, df: pd.DataFrame,
                        is_training: bool = True) -> pd.DataFrame:
        """
        Complete feature preparation pipeline.

        Args:
            df: DataFrame with raw demographic features
            is_training: Whether this is training data

        Returns:
            DataFrame with prepared features
        """
        logger.info(f"Preparing features (training={is_training})...")

        # Create engineered features
        df_features = self.create_engineered_features(df)

        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, is_training)

        # Scale numeric features
        df_scaled = self.scale_numeric_features(df_encoded, is_training)

        # Store feature columns (exclude target columns and customer_id)
        if is_training:
            exclude_cols = ['customer_id'] + [col for col in df_scaled.columns
                                             if col.startswith('has_')]
            self.feature_columns = [col for col in df_scaled.columns
                                   if col not in exclude_cols]
            logger.info(f"Total feature columns: {len(self.feature_columns)}")

        return df_scaled

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix from prepared DataFrame.

        Args:
            df: Prepared DataFrame

        Returns:
            NumPy array with features only
        """
        return df[self.feature_columns].values

    def get_target_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract target matrix (multi-label) from prepared DataFrame.

        Args:
            df: Prepared DataFrame

        Returns:
            NumPy array with binary plan indicators
        """
        target_cols = [col for col in df.columns if col.startswith('has_')]
        return df[target_cols].values

    def get_target_column_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get names of target columns.

        Args:
            df: DataFrame

        Returns:
            List of target column names
        """
        return [col for col in df.columns if col.startswith('has_')]

    def prepare_tier_features(self, df: pd.DataFrame,
                             is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for tier prediction.
        Uses the same feature columns as the main model.

        Args:
            df: DataFrame with demographics and Coverage Tier
            is_training: Whether this is training data (ignored - uses stored feature columns)

        Returns:
            Tuple of (X, y) where X is feature matrix and y is target array
        """
        # Create engineered features
        df_features = self.create_engineered_features(df)

        # Encode categorical features (always use stored columns from Stage 1)
        df_encoded = self.encode_categorical_features(df_features, is_training=False)

        # Scale numeric features (always use stored scaler from Stage 1)
        df_scaled = self.scale_numeric_features(df_encoded, is_training=False)

        # Ensure all required feature columns exist
        for col in self.feature_columns:
            if col not in df_scaled.columns:
                df_scaled[col] = 0

        # Get feature matrix
        X = df_scaled[self.feature_columns].values

        # Get target (Coverage Tier)
        if 'Coverage Tier' in df_scaled.columns:
            y = df_scaled['Coverage Tier'].values
        else:
            y = None

        return X, y

    def save(self, filepath: str = None) -> None:
        """
        Save the feature engineer state.

        Args:
            filepath: Path to save to (defaults to config.FEATURE_ENGINEERING_PATH)
        """
        if filepath is None:
            filepath = config.FEATURE_ENGINEERING_PATH

        state = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'one_hot_columns': self.one_hot_columns
        }

        joblib.dump(state, filepath)
        logger.info(f"Feature engineering state saved to {filepath}")

    def load(self, filepath: str = None) -> None:
        """
        Load the feature engineer state.

        Args:
            filepath: Path to load from (defaults to config.FEATURE_ENGINEERING_PATH)
        """
        if filepath is None:
            filepath = config.FEATURE_ENGINEERING_PATH

        state = joblib.load(filepath)

        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.feature_columns = state['feature_columns']
        self.one_hot_columns = state['one_hot_columns']

        logger.info(f"Feature engineering state loaded from {filepath}")

    def prepare_single_customer(self, customer_data: Dict) -> np.ndarray:
        """
        Prepare features for a single new customer.

        Args:
            customer_data: Dictionary with customer demographics
                          e.g., {'Sex': 'Male', 'Age as of Billing Start Date': 35,
                                'Estimated Annual Gross Income': 75000, ...}

        Returns:
            Feature array ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Prepare features (not training mode)
        df_prepared = self.prepare_features(df, is_training=False)

        # Get feature matrix
        X = self.get_feature_matrix(df_prepared)

        return X


if __name__ == '__main__':
    # Test feature engineering
    from src.data_processing import load_processed_data

    train_df, test_df, _ = load_processed_data()

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Prepare training features
    train_prepared = fe.prepare_features(train_df, is_training=True)
    X_train = fe.get_feature_matrix(train_prepared)
    y_train = fe.get_target_matrix(train_prepared)

    # Prepare test features
    test_prepared = fe.prepare_features(test_df, is_training=False)
    X_test = fe.get_feature_matrix(test_prepared)
    y_test = fe.get_target_matrix(test_prepared)

    # Save feature engineer
    fe.save()

    print("\n=== Feature Engineering Summary ===")
    print(f"Number of features: {len(fe.feature_columns)}")
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Target shape: {y_train.shape}")
