"""
Model evaluation and visualization module.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import print_evaluation_summary

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Handles model evaluation and visualization."""

    def __init__(self):
        """Initialize the evaluator."""
        self.stage1_metrics = None
        self.stage2_metrics = None

    def evaluate_stage1(self, stage1_model, X_test, y_test) -> Dict:
        """
        Evaluate Stage 1 model.

        Args:
            stage1_model: Trained Stage1Model instance
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating Stage 1 model...")
        self.stage1_metrics = stage1_model.evaluate(X_test, y_test)
        return self.stage1_metrics

    def evaluate_stage2(self, stage2_models, tier_datasets, feature_engineer) -> Dict:
        """
        Evaluate Stage 2 models.

        Args:
            stage2_models: Trained Stage2Models instance
            tier_datasets: Tier datasets with train/test splits
            feature_engineer: FeatureEngineer instance

        Returns:
            Dictionary of metrics per plan type
        """
        logger.info("Evaluating Stage 2 models...")
        self.stage2_metrics = stage2_models.evaluate_all_models(
            tier_datasets, feature_engineer
        )
        return self.stage2_metrics

    def plot_feature_importance(self, stage1_model, feature_columns,
                                top_n: int = 20, save_path: str = None) -> None:
        """
        Plot feature importance from Stage 1 model.

        Args:
            stage1_model: Trained Stage1Model instance
            feature_columns: List of feature names
            top_n: Number of top features to plot
            save_path: Path to save plot (optional)
        """
        logger.info("Plotting feature importance...")

        # Get feature importances
        importances = stage1_model.get_feature_importance(feature_columns, top_n)

        # Extract names and values
        features = [f[0] for f in importances]
        values = [f[1] for f in importances]

        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (Stage 1 Model)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels,
                             title: str = 'Confusion Matrix',
                             save_path: str = None) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
            save_path: Path to save plot (optional)
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_stage1_performance(self, save_path: str = None) -> None:
        """
        Plot Stage 1 per-label performance.

        Args:
            save_path: Path to save plot (optional)
        """
        if self.stage1_metrics is None:
            logger.error("Stage 1 metrics not available. Run evaluate_stage1 first.")
            return

        per_label = self.stage1_metrics['per_label_metrics']

        # Extract data
        plan_types = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for label, metrics in per_label.items():
            plan_name = label.replace('has_', '').replace('_', ' ')
            plan_types.append(plan_name)
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])

        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)[::-1][:15]  # Top 15

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(sorted_indices))
        width = 0.2

        ax.bar(x - 1.5*width, [accuracies[i] for i in sorted_indices],
               width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, [precisions[i] for i in sorted_indices],
               width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, [recalls[i] for i in sorted_indices],
               width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, [f1_scores[i] for i in sorted_indices],
               width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Plan Type')
        ax.set_ylabel('Score')
        ax.set_title('Stage 1: Per-Label Performance (Top 15 Plans)')
        ax.set_xticks(x)
        ax.set_xticklabels([plan_types[i] for i in sorted_indices],
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stage 1 performance plot saved to {save_path}")

        plt.show()

    def plot_stage2_performance(self, save_path: str = None) -> None:
        """
        Plot Stage 2 tier prediction accuracy.

        Args:
            save_path: Path to save plot (optional)
        """
        if self.stage2_metrics is None:
            logger.error("Stage 2 metrics not available. Run evaluate_stage2 first.")
            return

        # Extract data
        plan_types = []
        accuracies = []
        confidences = []
        n_samples = []

        for plan_type, metrics in self.stage2_metrics.items():
            plan_types.append(plan_type)
            accuracies.append(metrics['accuracy'])
            confidences.append(metrics['avg_confidence'])
            n_samples.append(metrics['n_samples'])

        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Accuracy
        x = np.arange(len(sorted_indices))
        ax1.barh(x, [accuracies[i] for i in sorted_indices], alpha=0.7)
        ax1.set_yticks(x)
        ax1.set_yticklabels([plan_types[i] for i in sorted_indices])
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Stage 2: Tier Prediction Accuracy by Plan Type')
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Sample size
        ax2.barh(x, [n_samples[i] for i in sorted_indices],
                alpha=0.7, color='orange')
        ax2.set_yticks(x)
        ax2.set_yticklabels([plan_types[i] for i in sorted_indices])
        ax2.set_xlabel('Number of Test Samples')
        ax2.set_title('Stage 2: Test Sample Size by Plan Type')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stage 2 performance plot saved to {save_path}")

        plt.show()

    def generate_evaluation_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Report as string
        """
        logger.info("Generating evaluation report...")

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("INSURANCE RECOMMENDATION ENGINE - EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")

        # Stage 1 Summary
        if self.stage1_metrics:
            report_lines.append("STAGE 1: MULTI-LABEL PLAN TYPE PREDICTION")
            report_lines.append("-"*80)
            report_lines.append(f"Hamming Loss: {self.stage1_metrics['hamming_loss']:.4f}")
            report_lines.append(f"Average Accuracy: {self.stage1_metrics['avg_accuracy']:.2%}")
            report_lines.append(f"Average Precision: {self.stage1_metrics['avg_precision']:.2%}")
            report_lines.append(f"Average Recall: {self.stage1_metrics['avg_recall']:.2%}")
            report_lines.append(f"Average F1-Score: {self.stage1_metrics['avg_f1']:.2%}")
            report_lines.append("")

            # Per-label details
            report_lines.append("Per-Label Performance:")
            report_lines.append(f"{'Plan Type':<40} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Supp':>8}")
            report_lines.append("-"*80)

            per_label = self.stage1_metrics['per_label_metrics']
            for label, metrics in sorted(per_label.items(),
                                        key=lambda x: x[1]['f1_score'],
                                        reverse=True):
                plan_name = label.replace('has_', '').replace('_', ' ')[:40]
                report_lines.append(
                    f"{plan_name:<40} "
                    f"{metrics['accuracy']:>8.2%} "
                    f"{metrics['precision']:>8.2%} "
                    f"{metrics['recall']:>8.2%} "
                    f"{metrics['f1_score']:>8.2%} "
                    f"{metrics['n_positive']:>8}"
                )

            report_lines.append("")

        # Stage 2 Summary
        if self.stage2_metrics:
            report_lines.append("STAGE 2: COVERAGE TIER PREDICTION")
            report_lines.append("-"*80)

            avg_accuracy = np.mean([m['accuracy'] for m in self.stage2_metrics.values()])
            report_lines.append(f"Average Tier Prediction Accuracy: {avg_accuracy:.2%}")
            report_lines.append(f"Number of Plan Types with Tier Models: {len(self.stage2_metrics)}")
            report_lines.append("")

            report_lines.append("Per-Plan Performance:")
            report_lines.append(f"{'Plan Type':<40} {'Accuracy':>12} {'Samples':>12}")
            report_lines.append("-"*80)

            for plan_type, metrics in sorted(self.stage2_metrics.items(),
                                            key=lambda x: x[1]['accuracy'],
                                            reverse=True):
                report_lines.append(
                    f"{plan_type:<40} "
                    f"{metrics['accuracy']:>12.2%} "
                    f"{metrics['n_samples']:>12}"
                )

        report_lines.append("")
        report_lines.append("="*80)

        report = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")

        return report


if __name__ == '__main__':
    # Example usage
    from src.data_processing import load_processed_data
    from src.feature_engineering import FeatureEngineer
    from src.stage1_model import Stage1Model
    from src.stage2_model import Stage2Models

    # Load data
    train_df, test_df, data_package = load_processed_data()
    tier_datasets = data_package['tier_train_test_splits']

    # Load models
    stage1_model = Stage1Model()
    stage1_model.load_model()

    stage2_models = Stage2Models()
    stage2_models.load_models()

    fe = FeatureEngineer()
    fe.load()

    # Prepare test data
    test_prepared = fe.prepare_features(test_df, is_training=False)
    X_test = fe.get_feature_matrix(test_prepared)
    y_test = fe.get_target_matrix(test_prepared)

    # Evaluate
    evaluator = ModelEvaluator()
    evaluator.evaluate_stage1(stage1_model, X_test, y_test)
    evaluator.evaluate_stage2(stage2_models, tier_datasets, fe)

    # Generate report
    report = evaluator.generate_evaluation_report('evaluation_report.txt')
    print(report)

    # Create visualizations
    evaluator.plot_feature_importance(stage1_model, fe.feature_columns)
    evaluator.plot_stage1_performance()
    evaluator.plot_stage2_performance()
