"""
Improved recommendation engine with comprehensive score-based output.
Shows recommendations for ALL plans with scores, tiers, and premiums.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.stage1_model import Stage1Model
from src.stage2_model import Stage2Models
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ImprovedRecommendationEngine:
    """
    Improved recommendation engine that provides comprehensive score-based recommendations
    for ALL insurance plans.
    """

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
        """Convert 'has_X' column name to plan name."""
        plan_name = has_column.replace('has_', '').replace('_', ' ')
        plan_name = plan_name.replace(' and ', ' & ')
        return plan_name

    def _estimate_premium(self, plan_type: str, coverage_tier: str) -> float:
        """Estimate premium for a plan and tier combination."""
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

    def _get_recommendation_level(self, score: float, threshold: float = None) -> str:
        """
        Get recommendation level based on score.

        Returns: "Highly Recommended", "Recommended", "Consider", "Not Recommended"
        """
        if threshold is None:
            threshold = config.CONFIDENCE_THRESHOLD

        if score >= 0.70:
            return "Highly Recommended"
        elif score >= 0.50:
            return "Recommended"
        elif score >= threshold:
            return "Consider"
        else:
            return "Not Recommended"

    def recommend_comprehensive(self, customer_data: Dict,
                               threshold: float = None) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations showing scores for ALL plans.

        Args:
            customer_data: Dictionary with customer demographics
            threshold: Minimum confidence threshold

        Returns:
            Dictionary with comprehensive plan recommendations
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")

        if threshold is None:
            threshold = config.CONFIDENCE_THRESHOLD

        logger.info("Generating comprehensive insurance recommendations...")

        # Prepare features
        X = self.feature_engineer.prepare_single_customer(customer_data)

        # Stage 1: Get scores for ALL plan types
        plan_probabilities = self.stage1_model.predict_proba(X)[0]

        all_recommendations = []

        # Process each plan type
        for i, target_col in enumerate(self.stage1_model.target_columns):
            plan_type = self._get_plan_name(target_col)
            score = plan_probabilities[i]

            # Get recommendation level
            rec_level = self._get_recommendation_level(score, threshold)

            # Predict coverage tier
            coverage_tier, tier_confidence = self.stage2_models.predict_tier(plan_type, X)

            # Estimate premium
            premium = self._estimate_premium(plan_type, coverage_tier)

            all_recommendations.append({
                'plan_type': plan_type,
                'score': score,
                'recommendation_level': rec_level,
                'coverage_tier': coverage_tier,
                'tier_confidence': tier_confidence,
                'estimated_premium': premium,
                'recommended': score >= threshold
            })

        # Sort by score (highest first)
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)

        # Split into recommended and not recommended
        recommended = [r for r in all_recommendations if r['recommended']]
        not_recommended = [r for r in all_recommendations if not r['recommended']]

        # Calculate total premium
        total_premium = sum([r['estimated_premium'] for r in recommended
                           if not np.isnan(r['estimated_premium'])])

        result = {
            'customer_data': customer_data,
            'all_plans': all_recommendations,
            'recommended_plans': recommended,
            'not_recommended': not_recommended,
            'total_plans_recommended': len(recommended),
            'confidence_threshold': threshold,
            'total_monthly_premium': total_premium,
            'total_annual_premium': total_premium * 12
        }

        logger.info(f"Generated recommendations: {len(recommended)} recommended, "
                   f"{len(not_recommended)} not recommended")

        return result


def print_comprehensive_recommendation(recommendations: Dict[str, Any]) -> None:
    """
    Print comprehensive recommendation in a clear, score-based format.

    Format matches user's expectations:
    - Binary recommendation for each plan
    - Coverage level
    - Score/confidence
    - Premium estimate
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE INSURANCE RECOMMENDATION")
    print("="*100)

    # Customer info
    customer = recommendations['customer_data']
    print(f"\nCUSTOMER PROFILE:")
    print(f"  Age: {customer['Age as of Billing Start Date']}")
    print(f"  Income: ${customer['Estimated Annual Gross Income']:,.0f}/year")
    print(f"  Sex: {customer['Sex']}")
    print(f"  State: {customer['State']}")
    print(f"  Job Class: {customer['Job Class']}")
    print(f"  Threshold: {recommendations['confidence_threshold']*100:.0f}%")

    print("\n" + "-"*100)
    print(f"{'PLAN TYPE':<35} {'SCORE':<10} {'RECOMMENDATION':<20} {'COVERAGE TIER':<25} {'PREMIUM':<15}")
    print("-"*100)

    # Show ALL plans sorted by score
    for plan in recommendations['all_plans']:
        plan_name = plan['plan_type']
        score = plan['score'] * 100
        rec_level = plan['recommendation_level']
        tier = plan['coverage_tier']

        # Format premium
        if np.isnan(plan['estimated_premium']):
            premium_str = "N/A"
        else:
            premium_str = f"${plan['estimated_premium']:.2f}/mo"

        # Color code recommendation
        if plan['recommended']:
            rec_symbol = "✓"
        else:
            rec_symbol = "✗"

        print(f"{rec_symbol} {plan_name:<33} {score:>6.1f}%   {rec_level:<20} {tier:<25} {premium_str:<15}")

    print("-"*100)

    # Summary
    print(f"\nRECOMMENDATION SUMMARY:")
    print(f"  Plans Recommended: {recommendations['total_plans_recommended']}")
    print(f"  Plans Not Recommended: {len(recommendations['not_recommended'])}")

    if recommendations['total_monthly_premium'] > 0:
        print(f"\n  Estimated Total Monthly Premium: ${recommendations['total_monthly_premium']:.2f}")
        print(f"  Estimated Total Annual Premium: ${recommendations['total_annual_premium']:.2f}")

    print("\n" + "="*100 + "\n")

    # Legend
    print("LEGEND:")
    print("  ✓ = Recommended (score >= threshold)")
    print("  ✗ = Not Recommended (score < threshold)")
    print("\n  Highly Recommended: Score >= 70%")
    print("  Recommended: Score >= 50%")
    print("  Consider: Score >= threshold (30%)")
    print("  Not Recommended: Score < threshold")
    print()


def create_example_customer(age: int = 35, income: float = 75000,
                           sex: str = 'Male', state: str = 'CA',
                           job_class: str = 'Professional') -> Dict:
    """Create an example customer dictionary."""
    return {
        'Sex': sex,
        'Age as of Billing Start Date': age,
        'Pay Frequency': 12,
        'Estimated Annual Gross Income': income,
        'State': state,
        'Employment Status': 'A',
        'Job Class': job_class
    }


if __name__ == '__main__':
    # Test the improved recommendation engine
    engine = ImprovedRecommendationEngine()

    try:
        engine.load_models()

        # Test customer
        customer = create_example_customer(40, 100000, 'Male', 'TX', 'Agent')

        print("\nTest Customer:")
        print(f"  Age: 40, Income: $100,000, Male, Texas, Agent")

        # Generate comprehensive recommendation
        rec = engine.recommend_comprehensive(customer)
        print_comprehensive_recommendation(rec)

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure models are trained first: python main.py train")
