"""
Data processing module for the Insurance Recommendation Engine.
Handles loading, cleaning, restructuring, and splitting data.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import (
    setup_directories,
    identify_unique_customers,
    create_multi_hot_encoding,
    create_tier_datasets,
    calculate_premium_estimates
)

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data processing operations."""

    def __init__(self):
        """Initialize the DataProcessor."""
        setup_directories()
        self.df_raw = None
        self.df_customers = None
        self.tier_datasets = None
        self.premium_estimates = None

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the source CSV file.

        Args:
            file_path: Path to the CSV file (defaults to config.SOURCE_DATA)

        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            file_path = config.SOURCE_DATA

        logger.info(f"Loading data from {file_path}")
        self.df_raw = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns")

        return self.df_raw

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data: handle missing values, remove outliers, filter invalid entries.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        initial_rows = len(df)

        # Make a copy
        df_clean = df.copy()

        # Remove rows with missing critical demographic information
        critical_cols = ['Sex', 'Age as of Billing Start Date',
                        'Estimated Annual Gross Income', 'State',
                        'Job Class', 'Plan Type']

        df_clean = df_clean.dropna(subset=critical_cols)
        logger.info(f"Removed {initial_rows - len(df_clean)} rows with missing critical data")

        # Remove outliers in age (keep 1st to 99th percentile)
        age_col = 'Age as of Billing Start Date'
        age_low = df_clean[age_col].quantile(config.OUTLIER_PERCENTILE_LOW / 100)
        age_high = df_clean[age_col].quantile(config.OUTLIER_PERCENTILE_HIGH / 100)
        df_clean = df_clean[
            (df_clean[age_col] >= age_low) &
            (df_clean[age_col] <= age_high)
        ]
        logger.info(f"Removed age outliers (kept {age_low:.0f} to {age_high:.0f})")

        # Remove outliers in income (keep 1st to 99th percentile)
        income_col = 'Estimated Annual Gross Income'
        income_low = df_clean[income_col].quantile(config.OUTLIER_PERCENTILE_LOW / 100)
        income_high = df_clean[income_col].quantile(config.OUTLIER_PERCENTILE_HIGH / 100)
        df_clean = df_clean[
            (df_clean[income_col] >= income_low) &
            (df_clean[income_col] <= income_high)
        ]
        logger.info(f"Removed income outliers (kept ${income_low:,.0f} to ${income_high:,.0f})")

        # Filter to only valid coverage tiers (remove "Declined" and empty)
        if 'Coverage Tier' in df_clean.columns:
            df_clean['Coverage Tier'] = df_clean['Coverage Tier'].fillna('')

        logger.info(f"Cleaning complete. Final row count: {len(df_clean)}")

        return df_clean

    def restructure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restructure data from multiple rows per customer to one row per customer
        with multi-hot encoding for plan types.

        Args:
            df: Cleaned DataFrame

        Returns:
            Restructured DataFrame with one row per customer
        """
        logger.info("Restructuring data to one row per customer...")

        # Identify unique customers
        df_with_ids = identify_unique_customers(df)

        # Create multi-hot encoding for plan types
        self.df_customers = create_multi_hot_encoding(df_with_ids, config.PLAN_TYPES)

        logger.info(f"Identified {len(self.df_customers)} unique customers")

        # Create tier datasets for Stage 2
        self.tier_datasets = create_tier_datasets(df_with_ids, config.PLAN_TYPES)
        logger.info(f"Created tier datasets for {len(self.tier_datasets)} plan types")

        # Calculate premium estimates
        self.premium_estimates = calculate_premium_estimates(df_with_ids, config.PLAN_TYPES)

        return self.df_customers

    def split_data(self, df: pd.DataFrame, test_size: float = None,
                   random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Args:
            df: DataFrame to split
            test_size: Proportion of test set (defaults to config.TEST_SIZE)
            random_state: Random seed (defaults to config.RANDOM_STATE)

        Returns:
            Tuple of (train_df, test_df)
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        if random_state is None:
            random_state = config.RANDOM_STATE

        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        logger.info(f"Train set: {len(train_df)} customers")
        logger.info(f"Test set: {len(test_df)} customers")

        return train_df, test_df

    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Save processed data to disk.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
        """
        logger.info("Saving processed data...")

        # Save train and test sets
        joblib.dump(train_df, config.TRAIN_DATA)
        joblib.dump(test_df, config.TEST_DATA)

        # Save tier datasets and premium estimates
        data_package = {
            'tier_datasets': self.tier_datasets,
            'premium_estimates': self.premium_estimates,
            'tier_train_test_splits': {}
        }

        # Split tier datasets into train/test
        for plan_type, tier_df in self.tier_datasets.items():
            if len(tier_df) >= config.MIN_SAMPLES_FOR_TIER_MODEL:
                tier_train, tier_test = train_test_split(
                    tier_df,
                    test_size=config.TEST_SIZE,
                    random_state=config.RANDOM_STATE
                )
                data_package['tier_train_test_splits'][plan_type] = {
                    'train': tier_train,
                    'test': tier_test
                }

        joblib.dump(data_package, config.PROCESSED_DATA)

        logger.info("Processed data saved successfully")

    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete data processing pipeline.

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Starting complete data processing pipeline...")

        # Load data
        df_raw = self.load_data()

        # Clean data
        df_clean = self.clean_data(df_raw)

        # Restructure data
        df_customers = self.restructure_data(df_clean)

        # Split data
        train_df, test_df = self.split_data(df_customers)

        # Save processed data
        self.save_processed_data(train_df, test_df)

        logger.info("Data processing pipeline complete!")

        return train_df, test_df


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load processed data from disk.

    Returns:
        Tuple of (train_df, test_df, data_package)
    """
    logger.info("Loading processed data from disk...")

    train_df = joblib.load(config.TRAIN_DATA)
    test_df = joblib.load(config.TEST_DATA)
    data_package = joblib.load(config.PROCESSED_DATA)

    logger.info(f"Loaded train set: {len(train_df)} customers")
    logger.info(f"Loaded test set: {len(test_df)} customers")

    return train_df, test_df, data_package


if __name__ == '__main__':
    # Run the data processing pipeline
    processor = DataProcessor()
    train_df, test_df = processor.process_all()

    print("\n=== Data Processing Summary ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"\nPlan type columns:")
    plan_cols = [col for col in train_df.columns if col.startswith('has_')]
    for col in plan_cols:
        print(f"  {col}: {train_df[col].sum()} customers have this plan")
