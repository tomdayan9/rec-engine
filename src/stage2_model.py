"""
Stage 2 Model: Coverage Tier Prediction
Predicts the appropriate coverage tier for each plan type.
"""
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Dict, Tuple, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class Stage2Models:
    """Collection of tier prediction models, one per plan type."""

    def __init__(self):
        """Initialize the Stage 2 models."""
        self.models = {}  # Dictionary of {plan_type: model}
        self.plan_types = []

    def train_all_models(self, tier_datasets: Dict[str, Dict],
                        feature_engineer) -> None:
        """
        Train tier prediction models for all plan types.

        Args:
            tier_datasets: Dictionary with train/test splits for each plan
            feature_engineer: FeatureEngineer instance for feature preparation
        """
        logger.info("Training Stage 2 models (tier prediction)...")

        for plan_type, data_splits in tier_datasets.items():
            logger.info(f"\nTraining tier model for: {plan_type}")

            train_data = data_splits['train']
            n_samples = len(train_data)

            if n_samples < config.MIN_SAMPLES_FOR_TIER_MODEL:
                logger.warning(f"Insufficient samples ({n_samples}) for {plan_type}. Skipping.")
                continue

            # Prepare features
            X_train, y_train = feature_engineer.prepare_tier_features(
                train_data, is_training=True
            )

            # Check if we have multiple tiers
            unique_tiers = np.unique(y_train)
            if len(unique_tiers) < 2:
                logger.warning(f"Only one tier for {plan_type}: {unique_tiers[0]}. Skipping.")
                continue

            # Train model
            model = RandomForestClassifier(**config.STAGE2_PARAMS)
            model.fit(X_train, y_train)

            self.models[plan_type] = model
            self.plan_types.append(plan_type)

            logger.info(f"Trained {plan_type} model with {n_samples} samples")
            logger.info(f"  Tiers: {list(unique_tiers)}")

        logger.info(f"\nStage 2 training complete. Trained {len(self.models)} tier models.")

    def predict_tier(self, plan_type: str, X: np.ndarray) -> Tuple[str, float]:
        """
        Predict coverage tier for a specific plan type.

        Args:
            plan_type: Name of the plan type
            X: Features (can be single sample or batch)

        Returns:
            Tuple of (predicted_tier, confidence)
        """
        if plan_type not in self.models:
            # Default to most common tier if no model exists
            logger.warning(f"No tier model for {plan_type}. Defaulting to 'Employee Only'.")
            return 'Employee Only', 0.5

        model = self.models[plan_type]

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predict
        tier = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = proba.max()

        return tier, confidence

    def predict_tier_batch(self, plan_type: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict coverage tiers for multiple samples.

        Args:
            plan_type: Name of the plan type
            X: Features (n_samples, n_features)

        Returns:
            Tuple of (predicted_tiers, confidences)
        """
        if plan_type not in self.models:
            # Default to most common tier if no model exists
            n_samples = X.shape[0]
            return np.array(['Employee Only'] * n_samples), np.array([0.5] * n_samples)

        model = self.models[plan_type]

        # Predict
        tiers = model.predict(X)
        probas = model.predict_proba(X)
        confidences = probas.max(axis=1)

        return tiers, confidences

    def evaluate_all_models(self, tier_datasets: Dict[str, Dict],
                           feature_engineer) -> Dict:
        """
        Evaluate all tier prediction models.

        Args:
            tier_datasets: Dictionary with train/test splits for each plan
            feature_engineer: FeatureEngineer instance

        Returns:
            Dictionary of evaluation metrics per plan type
        """
        logger.info("Evaluating Stage 2 models...")

        all_metrics = {}

        for plan_type in self.plan_types:
            if plan_type not in tier_datasets:
                continue

            test_data = tier_datasets[plan_type]['test']

            # Prepare features
            X_test, y_test = feature_engineer.prepare_tier_features(
                test_data, is_training=False
            )

            # Predict
            y_pred, confidences = self.predict_tier_batch(plan_type, X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Get classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            all_metrics[plan_type] = {
                'accuracy': accuracy,
                'n_samples': len(y_test),
                'confusion_matrix': conf_matrix,
                'classification_report': report,
                'avg_confidence': confidences.mean()
            }

            logger.info(f"{plan_type}: Accuracy = {accuracy:.2%} ({len(y_test)} samples)")

        return all_metrics

    def get_available_plan_types(self) -> List[str]:
        """
        Get list of plan types with trained models.

        Returns:
            List of plan type names
        """
        return self.plan_types

    def save_models(self, filepath_prefix: str = None) -> None:
        """
        Save all trained tier models.

        Args:
            filepath_prefix: Path prefix (defaults to config.STAGE2_MODEL_PREFIX)
        """
        if filepath_prefix is None:
            filepath_prefix = config.STAGE2_MODEL_PREFIX

        models_data = {
            'models': self.models,
            'plan_types': self.plan_types
        }

        filepath = f"{filepath_prefix}_tier_predictors.pkl"
        joblib.dump(models_data, filepath)
        logger.info(f"Stage 2 models saved to {filepath}")

    def load_models(self, filepath_prefix: str = None) -> None:
        """
        Load trained tier models.

        Args:
            filepath_prefix: Path prefix (defaults to config.STAGE2_MODEL_PREFIX)
        """
        if filepath_prefix is None:
            filepath_prefix = config.STAGE2_MODEL_PREFIX

        filepath = f"{filepath_prefix}_tier_predictors.pkl"
        models_data = joblib.load(filepath)

        self.models = models_data['models']
        self.plan_types = models_data['plan_types']

        logger.info(f"Stage 2 models loaded from {filepath}")
        logger.info(f"Loaded {len(self.models)} tier prediction models")

    def print_evaluation_summary(self, metrics: Dict) -> None:
        """
        Print detailed evaluation summary.

        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n" + "="*80)
        print("STAGE 2: TIER PREDICTION PERFORMANCE")
        print("="*80)

        # Sort by accuracy
        sorted_metrics = sorted(metrics.items(),
                               key=lambda x: x[1]['accuracy'],
                               reverse=True)

        print(f"\n{'Plan Type':<40} {'Accuracy':>12} {'Avg Conf':>12} {'Samples':>12}")
        print("-" * 80)

        for plan_type, scores in sorted_metrics:
            print(f"{plan_type:<40} "
                  f"{scores['accuracy']:>12.2%} "
                  f"{scores['avg_confidence']:>12.2%} "
                  f"{scores['n_samples']:>12}")

        print("="*80)

        # Print detailed report for each plan
        for plan_type, scores in sorted_metrics:
            print(f"\n{plan_type}:")
            print("-" * 80)
            report = scores['classification_report']

            # Print per-tier metrics
            for tier, tier_metrics in report.items():
                if tier in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                if isinstance(tier_metrics, dict):
                    print(f"  {tier}:")
                    print(f"    Precision: {tier_metrics['precision']:.2%}")
                    print(f"    Recall: {tier_metrics['recall']:.2%}")
                    print(f"    F1-Score: {tier_metrics['f1-score']:.2%}")
                    print(f"    Support: {tier_metrics['support']}")

        print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    # Test Stage 2 models
    from src.data_processing import load_processed_data
    from src.feature_engineering import FeatureEngineer

    # Load data
    train_df, test_df, data_package = load_processed_data()
    tier_datasets = data_package['tier_train_test_splits']

    # Prepare feature engineer
    fe = FeatureEngineer()
    fe.load()  # Load previously saved feature engineering state

    # Train models
    models = Stage2Models()
    models.train_all_models(tier_datasets, fe)

    # Evaluate
    metrics = models.evaluate_all_models(tier_datasets, fe)
    models.print_evaluation_summary(metrics)

    # Save models
    models.save_models()
