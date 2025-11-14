"""
Main entry point for the Insurance Recommendation Engine.
Provides CLI interface for training, evaluation, and prediction.
"""
import argparse
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processing import DataProcessor, load_processed_data
from src.feature_engineering import FeatureEngineer
from src.stage1_model import Stage1Model
from src.stage2_model import Stage2Models
from src.recommendation import InsuranceRecommendationEngine, create_example_customer
from src.recommendation_v2 import ImprovedRecommendationEngine, print_comprehensive_recommendation
from src.evaluation import ModelEvaluator
from src.utils import print_recommendation_summary, print_evaluation_summary

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def train_models():
    """Train all models (Stage 1 and Stage 2)."""
    print("\n" + "="*80)
    print("TRAINING INSURANCE RECOMMENDATION MODELS")
    print("="*80 + "\n")

    # Step 1: Process data
    print("Step 1: Processing data...")
    processor = DataProcessor()
    train_df, test_df = processor.process_all()
    print(f"✓ Data processed: {len(train_df)} training samples, {len(test_df)} test samples\n")

    # Load processed data
    train_df, test_df, data_package = load_processed_data()
    tier_datasets = data_package['tier_train_test_splits']

    # Step 2: Feature engineering
    print("Step 2: Engineering features...")
    fe = FeatureEngineer()
    train_prepared = fe.prepare_features(train_df, is_training=True)
    test_prepared = fe.prepare_features(test_df, is_training=False)

    X_train = fe.get_feature_matrix(train_prepared)
    y_train = fe.get_target_matrix(train_prepared)
    X_test = fe.get_feature_matrix(test_prepared)
    y_test = fe.get_target_matrix(test_prepared)

    target_columns = fe.get_target_column_names(train_prepared)

    fe.save()
    print(f"✓ Features engineered: {X_train.shape[1]} features\n")

    # Step 3: Train Stage 1 model
    print("Step 3: Training Stage 1 model (plan type prediction)...")
    stage1_model = Stage1Model()
    stage1_model.train(X_train, y_train, target_columns)
    stage1_model.save_model()
    print("✓ Stage 1 model trained and saved\n")

    # Step 4: Train Stage 2 models
    print("Step 4: Training Stage 2 models (tier prediction)...")
    stage2_models = Stage2Models()
    stage2_models.train_all_models(tier_datasets, fe)
    stage2_models.save_models()
    print("✓ Stage 2 models trained and saved\n")

    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")


def evaluate_models():
    """Evaluate trained models on test set."""
    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80 + "\n")

    # Load data
    train_df, test_df, data_package = load_processed_data()
    tier_datasets = data_package['tier_train_test_splits']

    # Load feature engineer
    fe = FeatureEngineer()
    fe.load()

    # Prepare test data
    test_prepared = fe.prepare_features(test_df, is_training=False)
    X_test = fe.get_feature_matrix(test_prepared)
    y_test = fe.get_target_matrix(test_prepared)

    # Load models
    stage1_model = Stage1Model()
    stage1_model.load_model()

    stage2_models = Stage2Models()
    stage2_models.load_models()

    # Evaluate
    evaluator = ModelEvaluator()

    print("Evaluating Stage 1 model...")
    stage1_metrics = evaluator.evaluate_stage1(stage1_model, X_test, y_test)
    stage1_model.print_per_label_performance(stage1_metrics)

    print("\nEvaluating Stage 2 models...")
    stage2_metrics = evaluator.evaluate_stage2(stage2_models, tier_datasets, fe)
    stage2_models.print_evaluation_summary(stage2_metrics)

    # Print summary
    print_evaluation_summary(stage1_metrics, stage2_metrics)

    # Generate report
    report_path = os.path.join(config.BASE_DIR, 'evaluation_report.txt')
    evaluator.generate_evaluation_report(report_path)
    print(f"\n✓ Detailed evaluation report saved to: {report_path}")

    # Show feature importance
    print("\nTop 20 Most Important Features:")
    print("-" * 80)
    for feature, importance in stage1_model.get_feature_importance(fe.feature_columns, 20):
        print(f"  {feature:<50} {importance:.4f}")

    print("\n" + "="*80 + "\n")


def predict(args):
    """Make recommendation for a new customer."""
    # Create customer data from arguments
    customer_data = {
        'Sex': args.sex,
        'Age as of Billing Start Date': args.age,
        'Pay Frequency': args.pay_frequency,
        'Estimated Annual Gross Income': args.income,
        'State': args.state,
        'Employment Status': args.employment_status,
        'Job Class': args.job_class
    }

    # Load and run IMPROVED recommendation engine
    engine = ImprovedRecommendationEngine()
    engine.load_models()

    # Generate comprehensive recommendation
    recommendation = engine.recommend_comprehensive(customer_data, args.confidence)

    # Print comprehensive results
    print_comprehensive_recommendation(recommendation)


def demo():
    """Run demonstration with example customers."""
    print("\n" + "="*80)
    print("DEMONSTRATION MODE - Example Recommendations")
    print("="*80 + "\n")

    # Load IMPROVED engine
    engine = ImprovedRecommendationEngine()
    engine.load_models()

    # Example customers
    examples = [
        {
            'name': 'Young Professional',
            'data': create_example_customer(28, 55000, 'Male', 'CA', 'Professional')
        },
        {
            'name': 'Mid-Career Agent',
            'data': create_example_customer(45, 150000, 'Female', 'NY', 'Agent')
        },
        {
            'name': 'Senior Executive',
            'data': create_example_customer(55, 300000, 'Male', 'TX', 'Agent')
        },
    ]

    for example in examples:
        print(f"\n{'='*100}")
        print(f"EXAMPLE {examples.index(example) + 1}: {example['name']}")
        print(f"{'='*100}")

        recommendation = engine.recommend_comprehensive(example['data'])
        print_comprehensive_recommendation(recommendation)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Insurance Recommendation Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train all models')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict for new customer')
    predict_parser.add_argument('--age', type=int, required=True,
                              help='Customer age')
    predict_parser.add_argument('--income', type=float, required=True,
                              help='Annual gross income')
    predict_parser.add_argument('--sex', type=str, required=True,
                              choices=['Male', 'Female'],
                              help='Sex')
    predict_parser.add_argument('--state', type=str, required=True,
                              help='State abbreviation (e.g., CA, NY)')
    predict_parser.add_argument('--job', dest='job_class', type=str,
                              default='Professional',
                              help='Job class (default: Professional)')
    predict_parser.add_argument('--employment-status', type=str,
                              default='A',
                              help='Employment status (default: A for Active)')
    predict_parser.add_argument('--pay-frequency', type=int,
                              default=12,
                              help='Pay frequency (default: 12 for monthly)')
    predict_parser.add_argument('--confidence', type=float,
                              default=config.CONFIDENCE_THRESHOLD,
                              help=f'Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration with example customers')

    args = parser.parse_args()

    if args.command == 'train':
        train_models()
    elif args.command == 'evaluate':
        evaluate_models()
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'demo':
        demo()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
