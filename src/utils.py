"""
Utility functions for the Insurance Recommendation Engine
"""
import logging
import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    logger.info("Directories setup complete")


def identify_unique_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify unique customers based on demographics.
    Since there's no customer ID, we group by demographic features.

    Args:
        df: DataFrame with multiple rows per customer

    Returns:
        DataFrame with unique customer identifiers
    """
    # Create a unique customer identifier based on demographics
    customer_id_cols = ['Sex', 'Age as of Billing Start Date',
                       'Estimated Annual Gross Income', 'State',
                       'Employment Status', 'Job Class']

    # Create customer ID by combining values
    df['customer_id'] = df[customer_id_cols].astype(str).agg('_'.join, axis=1)

    return df


def create_multi_hot_encoding(df: pd.DataFrame, plan_types: List[str]) -> pd.DataFrame:
    """
    Convert multiple rows per customer to single row with multi-hot encoding.

    Args:
        df: DataFrame with customer_id and Plan Type columns
        plan_types: List of all possible plan types

    Returns:
        DataFrame with one row per customer and binary columns for each plan
    """
    # Get unique customers with their demographics
    demographic_cols = ['customer_id', 'Sex', 'Age as of Billing Start Date',
                       'Pay Frequency', 'Estimated Annual Gross Income', 'State',
                       'Employment Status', 'Job Class']

    customers_df = df[demographic_cols].drop_duplicates(subset=['customer_id'])

    # Create binary columns for each plan type
    for plan_type in plan_types:
        plan_col_name = f'has_{plan_type.replace(" ", "_").replace("&", "and")}'
        # Check if customer has this plan
        customers_with_plan = df[df['Plan Type'] == plan_type]['customer_id'].unique()
        customers_df[plan_col_name] = customers_df['customer_id'].isin(customers_with_plan).astype(int)

    return customers_df


def create_tier_datasets(df: pd.DataFrame, plan_types: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Create separate datasets for tier prediction for each plan type.
    Only includes customers who have that specific plan.

    Args:
        df: Original DataFrame with all rows
        plan_types: List of all possible plan types

    Returns:
        Dictionary mapping plan type to its tier prediction dataset
    """
    tier_datasets = {}

    for plan_type in plan_types:
        # Filter to only customers with this plan type
        plan_df = df[df['Plan Type'] == plan_type].copy()

        if len(plan_df) < config.MIN_SAMPLES_FOR_TIER_MODEL:
            logger.warning(f"Plan type '{plan_type}' has only {len(plan_df)} samples. Skipping tier model.")
            continue

        # Select relevant columns
        demographic_cols = ['Sex', 'Age as of Billing Start Date', 'Pay Frequency',
                          'Estimated Annual Gross Income', 'State',
                          'Employment Status', 'Job Class']

        tier_df = plan_df[demographic_cols + ['Coverage Tier']].copy()

        # Remove rows with missing coverage tier
        tier_df = tier_df.dropna(subset=['Coverage Tier'])

        # Filter to valid coverage tiers only
        valid_tiers = config.COVERAGE_TIERS
        tier_df = tier_df[tier_df['Coverage Tier'].isin(valid_tiers)]

        if len(tier_df) >= config.MIN_SAMPLES_FOR_TIER_MODEL:
            tier_datasets[plan_type] = tier_df
            logger.info(f"Created tier dataset for '{plan_type}' with {len(tier_df)} samples")
        else:
            logger.warning(f"Plan type '{plan_type}' has only {len(tier_df)} valid samples after filtering.")

    return tier_datasets


def calculate_premium_estimates(df: pd.DataFrame, plan_types: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average premium for each plan type and coverage tier combination.

    Args:
        df: DataFrame with Plan Type, Coverage Tier, and Monthly Premium
        plan_types: List of all possible plan types

    Returns:
        Nested dictionary: {plan_type: {coverage_tier: avg_premium}}
    """
    premium_estimates = {}

    for plan_type in plan_types:
        plan_df = df[df['Plan Type'] == plan_type].copy()

        if len(plan_df) == 0:
            continue

        tier_premiums = {}
        for tier in config.COVERAGE_TIERS:
            tier_df = plan_df[plan_df['Coverage Tier'] == tier]
            if len(tier_df) > 0:
                # Convert premium to numeric, handling any string formatting
                premiums = pd.to_numeric(tier_df['Monthly Premium'], errors='coerce')
                avg_premium = premiums.mean()
                if not np.isnan(avg_premium):
                    tier_premiums[tier] = avg_premium

        if tier_premiums:
            premium_estimates[plan_type] = tier_premiums

    return premium_estimates


def format_premium(amount: float) -> str:
    """
    Format premium amount as currency string.

    Args:
        amount: Premium amount

    Returns:
        Formatted string like "$45.20/month"
    """
    if np.isnan(amount):
        return "N/A"
    return f"${amount:.2f}/month"


def print_recommendation_summary(recommendations: Dict[str, Any]) -> None:
    """
    Pretty print recommendation results.

    Args:
        recommendations: Dictionary containing recommended_plans and not_recommended
    """
    print("\n" + "="*80)
    print("INSURANCE PORTFOLIO RECOMMENDATION")
    print("="*80)

    print("\nRECOMMENDED PLANS:")
    print("-" * 80)

    if recommendations['recommended_plans']:
        for i, plan in enumerate(recommendations['recommended_plans'], 1):
            print(f"\n{i}. {plan['plan_type']}")
            print(f"   Coverage Tier: {plan['coverage_tier']}")
            print(f"   Confidence: {plan['confidence']:.1f}%")
            print(f"   Estimated Premium: {plan['estimated_premium']}")
    else:
        print("No plans recommended above confidence threshold.")

    print("\n" + "-" * 80)
    print("\nNOT RECOMMENDED:")
    print("-" * 80)

    if recommendations['not_recommended']:
        for plan in recommendations['not_recommended']:
            print(f"  - {plan['plan_type']} (Confidence: {plan['confidence']:.1f}%)")
    else:
        print("All plans recommended.")

    print("\n" + "="*80 + "\n")


def print_evaluation_summary(stage1_metrics: Dict, stage2_metrics: Dict) -> None:
    """
    Pretty print model evaluation results.

    Args:
        stage1_metrics: Dictionary with Stage 1 evaluation metrics
        stage2_metrics: Dictionary with Stage 2 evaluation metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)

    print("\nSTAGE 1: MULTI-LABEL PLAN TYPE PREDICTION")
    print("-" * 80)
    print(f"Overall Hamming Loss: {stage1_metrics.get('hamming_loss', 0):.4f}")
    print(f"Average Accuracy: {stage1_metrics.get('avg_accuracy', 0):.2%}")
    print(f"Average Precision: {stage1_metrics.get('avg_precision', 0):.2%}")
    print(f"Average Recall: {stage1_metrics.get('avg_recall', 0):.2%}")
    print(f"Average F1-Score: {stage1_metrics.get('avg_f1', 0):.2%}")

    print("\nSTAGE 2: COVERAGE TIER PREDICTION")
    print("-" * 80)
    if stage2_metrics:
        for plan_type, metrics in stage2_metrics.items():
            print(f"\n{plan_type}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
            print(f"  Samples: {metrics.get('n_samples', 0)}")
    else:
        print("No Stage 2 models evaluated.")

    print("\n" + "="*80 + "\n")
