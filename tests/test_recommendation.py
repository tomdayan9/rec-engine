"""
Unit tests for the Insurance Recommendation Engine.
"""
import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.recommendation import create_example_customer


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for the recommendation engine."""

    def test_create_example_customer(self):
        """Test customer data creation."""
        customer = create_example_customer(35, 75000, 'Male', 'CA', 'Professional')

        self.assertEqual(customer['Age as of Billing Start Date'], 35)
        self.assertEqual(customer['Estimated Annual Gross Income'], 75000)
        self.assertEqual(customer['Sex'], 'Male')
        self.assertEqual(customer['State'], 'CA')
        self.assertEqual(customer['Job Class'], 'Professional')
        self.assertEqual(customer['Pay Frequency'], 12)
        self.assertEqual(customer['Employment Status'], 'A')

    def test_customer_data_types(self):
        """Test that customer data has correct types."""
        customer = create_example_customer()

        self.assertIsInstance(customer['Age as of Billing Start Date'], int)
        self.assertIsInstance(customer['Estimated Annual Gross Income'], (int, float))
        self.assertIsInstance(customer['Sex'], str)
        self.assertIsInstance(customer['State'], str)
        self.assertIsInstance(customer['Job Class'], str)

    def test_config_values(self):
        """Test that config values are properly set."""
        self.assertIsNotNone(config.PLAN_TYPES)
        self.assertIsNotNone(config.COVERAGE_TIERS)
        self.assertGreater(len(config.PLAN_TYPES), 0)
        self.assertGreater(len(config.COVERAGE_TIERS), 0)
        self.assertIsInstance(config.CONFIDENCE_THRESHOLD, float)
        self.assertGreater(config.CONFIDENCE_THRESHOLD, 0)
        self.assertLess(config.CONFIDENCE_THRESHOLD, 1)


class TestDataStructures(unittest.TestCase):
    """Test data structure configurations."""

    def test_plan_types_list(self):
        """Test that plan types list is valid."""
        self.assertIn('Long Term Disability', config.PLAN_TYPES)
        self.assertIn('Short Term Disability', config.PLAN_TYPES)
        self.assertIn('Dental', config.PLAN_TYPES)
        self.assertIn('Vision', config.PLAN_TYPES)

    def test_coverage_tiers_list(self):
        """Test that coverage tiers list is valid."""
        self.assertIn('Employee Only', config.COVERAGE_TIERS)

    def test_model_parameters(self):
        """Test that model parameters are valid."""
        self.assertIsInstance(config.STAGE1_PARAMS, dict)
        self.assertIsInstance(config.STAGE2_PARAMS, dict)
        self.assertIn('n_estimators', config.STAGE1_PARAMS)
        self.assertIn('random_state', config.STAGE1_PARAMS)


if __name__ == '__main__':
    unittest.main()
