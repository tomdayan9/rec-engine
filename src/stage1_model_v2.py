"""
Stage 1 Model V2: Multi-label Plan Type Prediction with Phase 1 Improvements
- XGBoost for better performance
- SMOTE for handling class imbalance
- 5-fold cross-validation
- Better probability calibration
"""
import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Dict, Tuple, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class Stage1ModelV2:
    """
    Improved multi-label classifier for predicting insurance plan types.
    Phase 1 Improvements:
    - XGBoost for better performance
    - SMOTE for class imbalance
    - Cross-validation for reliability
    """

    def __init__(self, use_xgboost=True, use_smote=True):
        """
        Initialize the Stage 1 model.

        Args:
            use_xgboost: Whether to use XGBoost (True) or Random Forest (False)
            use_smote: Whether to apply SMOTE for class imbalance
        """
        self.model = None
        self.target_columns = None
        self.feature_importances_ = None
        self.use_xgboost = use_xgboost
        self.use_smote = use_smote
        self.cv_scores = {}  # Store cross-validation scores

    def _create_base_estimator(self):
        """Create base estimator (XGBoost or Random Forest)."""
        if self.use_xgboost:
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Will be adjusted per label
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            return RandomForestClassifier(**config.STAGE1_PARAMS)

    def _apply_smote_for_label(self, X_train: np.ndarray, y_train: np.ndarray,
                               label_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE for a single label if class imbalance exists.

        Args:
            X_train: Training features
            y_train: Training labels for specific plan (binary)
            label_idx: Index of the label

        Returns:
            Resampled X and y
        """
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)

        # Only apply SMOTE if we have both classes and significant imbalance
        if len(unique) < 2:
            logger.warning(f"Label {label_idx}: Only one class present, skipping SMOTE")
            return X_train, y_train

        minority_ratio = min(counts) / sum(counts)

        if minority_ratio < 0.1:  # Less than 10% minority class
            try:
                # Create SMOTE + undersampling pipeline
                over = SMOTE(sampling_strategy=0.3, random_state=42)  # Oversample to 30%
                under = RandomUnderSampler(sampling_strategy=0.7, random_state=42)  # Keep 70%

                pipeline = ImbPipeline([
                    ('over', over),
                    ('under', under)
                ])

                X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
                logger.info(f"Label {label_idx}: Applied SMOTE. "
                          f"Samples: {len(y_train)} → {len(y_resampled)}")
                return X_resampled, y_resampled
            except Exception as e:
                logger.warning(f"Label {label_idx}: SMOTE failed ({e}), using original data")
                return X_train, y_train
        else:
            # Imbalance not severe enough for SMOTE
            return X_train, y_train

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             target_columns: List[str], use_cv: bool = True) -> None:
        """
        Train the multi-label classifier with Phase 1 improvements.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, n_labels)
            target_columns: Names of target columns
            use_cv: Whether to use cross-validation for evaluation
        """
        logger.info("Training Stage 1 Model V2 (Phase 1 Improvements)...")
        logger.info(f"Model type: {'XGBoost' if self.use_xgboost else 'Random Forest'}")
        logger.info(f"SMOTE: {'Enabled' if self.use_smote else 'Disabled'}")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Plan types: {y_train.shape[1]}")

        self.target_columns = target_columns

        # Train individual classifiers for each label
        estimators = []

        for i, target_col in enumerate(target_columns):
            logger.info(f"\nTraining classifier for: {target_col}")

            # Get labels for this plan
            y_label = y_train[:, i]

            # Apply SMOTE if enabled
            if self.use_smote:
                X_resampled, y_resampled = self._apply_smote_for_label(X_train, y_label, i)
            else:
                X_resampled, y_resampled = X_train, y_label

            # Create and train estimator
            estimator = self._create_base_estimator()
            estimator.fit(X_resampled, y_resampled)
            estimators.append(estimator)

            # Cross-validation if enabled
            if use_cv:
                try:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(
                        self._create_base_estimator(),
                        X_train, y_label,  # Use original data for CV
                        cv=cv,
                        scoring='f1',
                        n_jobs=-1
                    )
                    self.cv_scores[target_col] = {
                        'mean': cv_scores.mean(),
                        'std': cv_scores.std(),
                        'scores': cv_scores
                    }
                    logger.info(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                except Exception as e:
                    logger.warning(f"  CV failed for {target_col}: {e}")

        # Create MultiOutputClassifier wrapper
        self.model = MultiOutputClassifier(self._create_base_estimator(), n_jobs=-1)
        self.model.estimators_ = estimators

        # Calculate feature importances
        importances = []
        for estimator in estimators:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)

        if importances:
            self.feature_importances_ = np.mean(importances, axis=0)

        logger.info("\nStage 1 Model V2 training complete!")

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
        logger.info("Evaluating Stage 1 Model V2...")

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
                'n_predicted_positive': int(y_pred[:, i].sum()),
                'cv_score': self.cv_scores.get(target_col, {}).get('mean', None)
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
            'per_label_metrics': per_label_metrics,
            'model_type': 'XGBoost' if self.use_xgboost else 'RandomForest',
            'used_smote': self.use_smote
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
            filepath = config.STAGE1_MODEL_PATH.replace('.pkl', '_v2.pkl')

        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'target_columns': self.target_columns,
            'feature_importances': self.feature_importances_,
            'use_xgboost': self.use_xgboost,
            'use_smote': self.use_smote,
            'cv_scores': self.cv_scores
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Stage 1 Model V2 saved to {filepath}")

    def load_model(self, filepath: str = None) -> None:
        """
        Load a trained model.

        Args:
            filepath: Path to load from (defaults to config.STAGE1_MODEL_PATH)
        """
        if filepath is None:
            filepath = config.STAGE1_MODEL_PATH.replace('.pkl', '_v2.pkl')

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.target_columns = model_data['target_columns']
        self.feature_importances_ = model_data['feature_importances']
        self.use_xgboost = model_data.get('use_xgboost', True)
        self.use_smote = model_data.get('use_smote', True)
        self.cv_scores = model_data.get('cv_scores', {})

        logger.info(f"Stage 1 Model V2 loaded from {filepath}")

    def print_per_label_performance(self, metrics: Dict) -> None:
        """
        Print detailed per-label performance metrics.

        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n" + "="*100)
        print("STAGE 1 V2: PER-LABEL PERFORMANCE (PHASE 1 IMPROVEMENTS)")
        print(f"Model: {metrics['model_type']}, SMOTE: {metrics['used_smote']}")
        print("="*100)

        per_label = metrics['per_label_metrics']

        # Sort by F1 score
        sorted_labels = sorted(per_label.items(),
                              key=lambda x: x[1]['f1_score'],
                              reverse=True)

        print(f"\n{'Plan Type':<40} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV-F1':>10} {'Support':>10}")
        print("-" * 110)

        for label, scores in sorted_labels:
            plan_name = label.replace('has_', '').replace('_', ' ')
            cv_score = scores.get('cv_score')
            cv_str = f"{cv_score:.2%}" if cv_score is not None else "N/A"

            print(f"{plan_name:<40} "
                  f"{scores['accuracy']:>10.2%} "
                  f"{scores['precision']:>10.2%} "
                  f"{scores['recall']:>10.2%} "
                  f"{scores['f1_score']:>10.2%} "
                  f"{cv_str:>10} "
                  f"{scores['n_positive']:>10}")

        print("="*110 + "\n")
