"""
Stage 1 Model: Multi-label Plan Type Prediction
Predicts which insurance plans a customer should have.
"""
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, classification_report
)
import joblib
from typing import Dict, Tuple, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class Stage1Model:
    """Multi-label classifier for predicting insurance plan types."""

    def __init__(self):
        """Initialize the Stage 1 model."""
        self.model = None
        self.target_columns = None
        self.feature_importances_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             target_columns: List[str]) -> None:
        """
        Train the multi-label classifier.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, n_labels)
            target_columns: Names of target columns
        """
        logger.info("Training Stage 1 model (multi-label plan prediction)...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Plan types: {y_train.shape[1]}")

        self.target_columns = target_columns

        # Create base estimator
        base_estimator = RandomForestClassifier(**config.STAGE1_PARAMS)

        # Create multi-output classifier
        self.model = MultiOutputClassifier(base_estimator, n_jobs=-1)

        # Train the model
        self.model.fit(X_train, y_train)

        # Store feature importances (average across all estimators)
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        self.feature_importances_ = np.mean(importances, axis=0)

        logger.info("Stage 1 model training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict plan types for new customers.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Binary predictions (n_samples, n_labels)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability scores for plan types.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probability scores (n_samples, n_labels)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get probabilities for positive class from each estimator
        probas = []
        for estimator in self.model.estimators_:
            # Get probability for class 1 (has the plan)
            proba = estimator.predict_proba(X)[:, 1] if len(estimator.classes_) > 1 else np.zeros(X.shape[0])
            probas.append(proba)

        # Stack to get (n_samples, n_labels)
        return np.column_stack(probas)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Stage 1 model...")

        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # Calculate overall metrics
        hamming = hamming_loss(y_test, y_pred)

        # Calculate per-label metrics
        per_label_metrics = {}
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for i, target_col in enumerate(self.target_columns):
            accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
            precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)

            per_label_metrics[target_col] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_positive': int(y_test[:, i].sum()),
                'n_predicted_positive': int(y_pred[:, i].sum())
            }

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        metrics = {
            'hamming_loss': hamming,
            'avg_accuracy': np.mean(accuracies),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_f1': np.mean(f1_scores),
            'per_label_metrics': per_label_metrics
        }

        logger.info(f"Hamming Loss: {hamming:.4f}")
        logger.info(f"Average Accuracy: {metrics['avg_accuracy']:.2%}")
        logger.info(f"Average Precision: {metrics['avg_precision']:.2%}")
        logger.info(f"Average Recall: {metrics['avg_recall']:.2%}")
        logger.info(f"Average F1-Score: {metrics['avg_f1']:.2%}")

        return metrics

    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top feature importances.

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained yet.")

        # Create list of (feature, importance) tuples
        feature_importance = list(zip(feature_names, self.feature_importances_))

        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance[:top_n]

    def save_model(self, filepath: str = None) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save to (defaults to config.STAGE1_MODEL_PATH)
        """
        if filepath is None:
            filepath = config.STAGE1_MODEL_PATH

        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'target_columns': self.target_columns,
            'feature_importances': self.feature_importances_
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Stage 1 model saved to {filepath}")

    def load_model(self, filepath: str = None) -> None:
        """
        Load a trained model.

        Args:
            filepath: Path to load from (defaults to config.STAGE1_MODEL_PATH)
        """
        if filepath is None:
            filepath = config.STAGE1_MODEL_PATH

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.target_columns = model_data['target_columns']
        self.feature_importances_ = model_data['feature_importances']

        logger.info(f"Stage 1 model loaded from {filepath}")

    def print_per_label_performance(self, metrics: Dict) -> None:
        """
        Print detailed per-label performance metrics.

        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n" + "="*80)
        print("STAGE 1: PER-LABEL PERFORMANCE")
        print("="*80)

        per_label = metrics['per_label_metrics']

        # Sort by F1 score
        sorted_labels = sorted(per_label.items(),
                              key=lambda x: x[1]['f1_score'],
                              reverse=True)

        print(f"\n{'Plan Type':<40} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 100)

        for label, scores in sorted_labels:
            plan_name = label.replace('has_', '').replace('_', ' ')
            print(f"{plan_name:<40} "
                  f"{scores['accuracy']:>10.2%} "
                  f"{scores['precision']:>10.2%} "
                  f"{scores['recall']:>10.2%} "
                  f"{scores['f1_score']:>10.2%} "
                  f"{scores['n_positive']:>10}")

        print("="*100 + "\n")


if __name__ == '__main__':
    # Test Stage 1 model
    from src.data_processing import load_processed_data
    from src.feature_engineering import FeatureEngineer

    # Load data
    train_df, test_df, _ = load_processed_data()

    # Prepare features
    fe = FeatureEngineer()
    train_prepared = fe.prepare_features(train_df, is_training=True)
    test_prepared = fe.prepare_features(test_df, is_training=False)

    X_train = fe.get_feature_matrix(train_prepared)
    y_train = fe.get_target_matrix(train_prepared)
    X_test = fe.get_feature_matrix(test_prepared)
    y_test = fe.get_target_matrix(test_prepared)

    target_columns = fe.get_target_column_names(train_prepared)

    # Train model
    model = Stage1Model()
    model.train(X_train, y_train, target_columns)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    model.print_per_label_performance(metrics)

    # Save model
    model.save_model()

    print("\nTop 20 Important Features:")
    for feature, importance in model.get_feature_importance(fe.feature_columns, 20):
        print(f"  {feature}: {importance:.4f}")
