"""
Training script for Phase 1 improvements.
Trains both old and new models for comparison.
"""
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processing import DataProcessor, load_processed_data
from src.feature_engineering import FeatureEngineer
from src.stage1_model import Stage1Model
from src.stage1_model_v2 import Stage1ModelV2

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    """Train and compare Phase 1 improvements."""
    print("\n" + "="*100)
    print("PHASE 1 IMPROVEMENTS - TRAINING AND COMPARISON")
    print("="*100 + "\n")

    # Step 1: Process data
    print("Step 1: Processing data...")
    try:
        train_df, test_df, data_package = load_processed_data()
        print(f"✓ Loaded processed data: {len(train_df)} train, {len(test_df)} test\n")
    except:
        print("Processing data from source...")
        processor = DataProcessor()
        train_df, test_df = processor.process_all()
        print(f"✓ Data processed: {len(train_df)} train, {len(test_df)} test\n")

    # Step 2: Feature engineering
    print("Step 2: Engineering features (with interaction features)...")
    fe = FeatureEngineer()
    train_prepared = fe.prepare_features(train_df, is_training=True)
    test_prepared = fe.prepare_features(test_df, is_training=False)

    X_train = fe.get_feature_matrix(train_prepared)
    y_train = fe.get_target_matrix(train_prepared)
    X_test = fe.get_feature_matrix(test_prepared)
    y_test = fe.get_target_matrix(test_prepared)

    target_columns = fe.get_target_column_names(train_prepared)

    fe.save()
    print(f"✓ Features engineered: {X_train.shape[1]} features (up from 47)\n")

    # Step 3: Train baseline model (Random Forest)
    print("="*100)
    print("BASELINE MODEL: Random Forest (Original)")
    print("="*100 + "\n")

    baseline_model = Stage1Model()
    baseline_model.train(X_train, y_train, target_columns)

    print("\nEvaluating baseline model...")
    baseline_metrics = baseline_model.evaluate(X_test, y_test)
    baseline_model.print_per_label_performance(baseline_metrics)

    # Step 4: Train Phase 1 model (XGBoost + SMOTE + CV)
    print("\n" + "="*100)
    print("PHASE 1 MODEL: XGBoost + SMOTE + Cross-Validation")
    print("="*100 + "\n")

    phase1_model = Stage1ModelV2(use_xgboost=True, use_smote=True)
    phase1_model.train(X_train, y_train, target_columns, use_cv=True)

    print("\nEvaluating Phase 1 model...")
    phase1_metrics = phase1_model.evaluate(X_test, y_test)
    phase1_model.print_per_label_performance(phase1_metrics)

    # Step 5: Save models
    print("\nSaving models...")
    baseline_model.save_model()
    phase1_model.save_model()
    print("✓ Both models saved\n")

    # Step 6: Comparison summary
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON")
    print("="*100 + "\n")

    print(f"{'Metric':<30} {'Baseline (RF)':<20} {'Phase 1 (XGBoost+SMOTE)':<25} {'Improvement':<15}")
    print("-"*100)

    metrics_to_compare = [
        ('Hamming Loss', 'hamming_loss', True),  # Lower is better
        ('Average Accuracy', 'avg_accuracy', False),
        ('Average Precision', 'avg_precision', False),
        ('Average Recall', 'avg_recall', False),
        ('Average F1-Score', 'avg_f1', False)
    ]

    for metric_name, metric_key, lower_is_better in metrics_to_compare:
        baseline_val = baseline_metrics[metric_key]
        phase1_val = phase1_metrics[metric_key]

        if lower_is_better:
            improvement = ((baseline_val - phase1_val) / baseline_val) * 100
            improvement_str = f"{improvement:+.2f}% better" if improvement > 0 else f"{abs(improvement):.2f}% worse"
        else:
            improvement = ((phase1_val - baseline_val) / baseline_val) * 100
            improvement_str = f"{improvement:+.2f}%" if improvement > 0 else f"{improvement:.2f}%"

        # Format values
        if metric_key == 'hamming_loss':
            baseline_str = f"{baseline_val:.4f}"
            phase1_str = f"{phase1_val:.4f}"
        else:
            baseline_str = f"{baseline_val:.2%}"
            phase1_str = f"{phase1_val:.2%}"

        print(f"{metric_name:<30} {baseline_str:<20} {phase1_str:<25} {improvement_str:<15}")

    print("\n" + "="*100)

    # Step 7: Top feature importances comparison
    print("\nTop 20 Most Important Features (Phase 1 Model):")
    print("-" * 100)
    for i, (feature, importance) in enumerate(phase1_model.get_feature_importance(fe.feature_columns, 20), 1):
        print(f"{i:2d}. {feature:<60} {importance:.4f}")

    print("\n" + "="*100)
    print("TRAINING COMPLETE!")
    print("="*100 + "\n")

    print("Next steps:")
    print("1. Review the performance improvements above")
    print("2. Use the Phase 1 model by default: import Stage1ModelV2")
    print("3. Update main.py to use Stage1ModelV2")
    print("4. Commit changes to GitHub\n")


if __name__ == '__main__':
    main()
