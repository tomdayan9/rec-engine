"""
Main recommendation engine that combines Stage 1 and Stage 2 models
to generate complete insurance portfolio recommendations.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.stage1_model import Stage1Model
from src.stage2_model import Stage2Models
from src.feature_engineering import FeatureEngineer
from src.utils import format_premium, print_recommendation_summary

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class InsuranceRecommendationEngine:
    """Complete insurance recommendation system."""

    def __init__(self):
        """Initialize the recommendation engine."""
        self.stage1_model = Stage1Model()
        self.stage2_models = Stage2Models()
        self.feature_engineer = FeatureEngineer()
        self.premium_estimates = None
        self.is_loaded = False

    def load_models(self) -> None:
        """Load all trained models and components."""
        logger.info("Loading recommendation engine components...")

        try:
            # Load Stage 1 model
            self.stage1_model.load_model()
            logger.info("Stage 1 model loaded")

            # Load Stage 2 models
            self.stage2_models.load_models()
            logger.info("Stage 2 models loaded")

            # Load feature engineer
            self.feature_engineer.load()
            logger.info("Feature engineer loaded")

            # Load premium estimates
            data_package = joblib.load(config.PROCESSED_DATA)
            self.premium_estimates = data_package['premium_estimates']
            logger.info("Premium estimates loaded")

            self.is_loaded = True
            logger.info("All components loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _get_plan_name(self, has_column: str) -> str:
        """
        Convert 'has_X' column name to plan name.

        Args:
            has_column: Column name like 'has_Long_Term_Disability'

        Returns:
            Plan name like 'Long Term Disability'
        """
        # Remove 'has_' prefix and replace underscores with spaces
        plan_name = has_column.replace('has_', '').replace('_', ' ')

        # Handle special case for 'and' -> '&'
        plan_name = plan_name.replace(' and ', ' & ')

        return plan_name

    def _estimate_premium(self, plan_type: str, coverage_tier: str) -> float:
        """
        Estimate premium for a plan and tier combination.

        Args:
            plan_type: Name of the plan
            coverage_tier: Coverage tier

        Returns:
            Estimated monthly premium
        """
        if self.premium_estimates is None:
            return np.nan

        if plan_type in self.premium_estimates:
            tier_premiums = self.premium_estimates[plan_type]
            if coverage_tier in tier_premiums:
                return tier_premiums[coverage_tier]
            else:
                # Return average premium for this plan if tier not found
                return np.mean(list(tier_premiums.values()))

        return np.nan

    def recommend(self, customer_data: Dict,
                 confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Generate complete insurance portfolio recommendation for a customer.

        Args:
            customer_data: Dictionary with customer demographics
                          Keys: 'Sex', 'Age as of Billing Start Date',
                               'Pay Frequency', 'Estimated Annual Gross Income',
                               'State', 'Employment Status', 'Job Class'
            confidence_threshold: Minimum confidence for recommendation (default from config)

        Returns:
            Dictionary with recommended_plans and not_recommended lists
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")

        if confidence_threshold is None:
            confidence_threshold = config.CONFIDENCE_THRESHOLD

        logger.info("Generating insurance recommendations...")
        logger.info(f"Customer: {customer_data}")

        # Prepare features
        X = self.feature_engineer.prepare_single_customer(customer_data)

        # Stage 1: Predict which plans to recommend
        plan_predictions = self.stage1_model.predict(X)[0]
        plan_probabilities = self.stage1_model.predict_proba(X)[0]

        recommended_plans = []
        not_recommended = []

        # For each plan type
        for i, target_col in enumerate(self.stage1_model.target_columns):
            plan_type = self._get_plan_name(target_col)
            confidence = plan_probabilities[i] * 100  # Convert to percentage

            # Stage 2: Predict coverage tier if plan is recommended
            if plan_predictions[i] == 1 and confidence >= (confidence_threshold * 100):
                # Predict tier
                coverage_tier, tier_confidence = self.stage2_models.predict_tier(plan_type, X)

                # Estimate premium
                premium = self._estimate_premium(plan_type, coverage_tier)

                recommended_plans.append({
                    'plan_type': plan_type,
                    'coverage_tier': coverage_tier,
                    'confidence': confidence,
                    'tier_confidence': tier_confidence * 100,
                    'estimated_premium': format_premium(premium)
                })
            else:
                not_recommended.append({
                    'plan_type': plan_type,
                    'confidence': confidence
                })

        # Sort recommended plans by confidence
        recommended_plans.sort(key=lambda x: x['confidence'], reverse=True)
        not_recommended.sort(key=lambda x: x['confidence'], reverse=True)

        result = {
            'customer_data': customer_data,
            'recommended_plans': recommended_plans,
            'not_recommended': not_recommended,
            'total_plans_recommended': len(recommended_plans),
            'confidence_threshold': confidence_threshold
        }

        logger.info(f"Recommended {len(recommended_plans)} plans")

        return result

    def recommend_batch(self, customers: List[Dict],
                       confidence_threshold: float = None) -> List[Dict]:
        """
        Generate recommendations for multiple customers.

        Args:
            customers: List of customer data dictionaries
            confidence_threshold: Minimum confidence for recommendation

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        for customer in customers:
            rec = self.recommend(customer, confidence_threshold)
            recommendations.append(rec)

        return recommendations

    def explain_recommendation(self, customer_data: Dict, plan_type: str) -> Dict:
        """
        Explain why a specific plan was or wasn't recommended.

        Args:
            customer_data: Customer demographics
            plan_type: Plan type to explain

        Returns:
            Dictionary with explanation details
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Prepare features
        X = self.feature_engineer.prepare_single_customer(customer_data)

        # Get prediction and probability for this plan
        plan_probabilities = self.stage1_model.predict_proba(X)[0]

        # Find the index for this plan type
        plan_col = f"has_{plan_type.replace(' ', '_').replace('&', 'and')}"

        try:
            plan_idx = self.stage1_model.target_columns.index(plan_col)
        except ValueError:
            return {
                'plan_type': plan_type,
                'error': 'Plan type not found in model'
            }

        confidence = plan_probabilities[plan_idx] * 100
        is_recommended = confidence >= (config.CONFIDENCE_THRESHOLD * 100)

        explanation = {
            'plan_type': plan_type,
            'is_recommended': is_recommended,
            'confidence': confidence,
            'threshold': config.CONFIDENCE_THRESHOLD * 100,
            'customer_data': customer_data
        }

        # Add tier prediction if recommended
        if is_recommended:
            coverage_tier, tier_confidence = self.stage2_models.predict_tier(plan_type, X)
            premium = self._estimate_premium(plan_type, coverage_tier)

            explanation['coverage_tier'] = coverage_tier
            explanation['tier_confidence'] = tier_confidence * 100
            explanation['estimated_premium'] = format_premium(premium)

        # Add feature importances for this plan
        if hasattr(self.stage1_model, 'feature_importances_'):
            top_features = self.stage1_model.get_feature_importance(
                self.feature_engineer.feature_columns, top_n=10
            )
            explanation['top_influencing_features'] = top_features

        return explanation


def create_example_customer(age: int = 35, income: float = 75000,
                           sex: str = 'Male', state: str = 'CA',
                           job_class: str = 'Professional') -> Dict:
    """
    Create an example customer dictionary.

    Args:
        age: Customer age
        income: Annual income
        sex: Sex (Male/Female)
        state: State abbreviation
        job_class: Job classification

    Returns:
        Customer data dictionary
    """
    return {
        'Sex': sex,
        'Age as of Billing Start Date': age,
        'Pay Frequency': 12,  # Monthly
        'Estimated Annual Gross Income': income,
        'State': state,
        'Employment Status': 'A',  # Active
        'Job Class': job_class
    }


if __name__ == '__main__':
    # Test the recommendation engine
    engine = InsuranceRecommendationEngine()

    try:
        engine.load_models()

        # Create example customers
        customers = [
            create_example_customer(35, 75000, 'Male', 'CA', 'Professional'),
            create_example_customer(45, 150000, 'Female', 'NY', 'Agent'),
            create_example_customer(28, 50000, 'Male', 'TX', 'Professional'),
        ]

        # Generate recommendations
        for i, customer in enumerate(customers, 1):
            print(f"\n{'='*80}")
            print(f"CUSTOMER {i}")
            print(f"{'='*80}")
            print(f"Age: {customer['Age as of Billing Start Date']}")
            print(f"Income: ${customer['Estimated Annual Gross Income']:,}")
            print(f"Sex: {customer['Sex']}")
            print(f"State: {customer['State']}")
            print(f"Job: {customer['Job Class']}")

            rec = engine.recommend(customer)
            print_recommendation_summary(rec)

    except Exception as e:
        print(f"Error: Models not trained yet. Please run training first.")
        print(f"Details: {e}")
