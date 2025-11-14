"""
Configuration settings for the Insurance Recommendation Engine
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SOURCE_DATA = os.path.join(BASE_DIR, 'source.csv')
PROCESSED_DATA = os.path.join(DATA_DIR, 'processed_data.pkl')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_data.pkl')
TEST_DATA = os.path.join(DATA_DIR, 'test_data.pkl')

# Model paths
STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, 'stage1_plan_predictor.pkl')
STAGE2_MODEL_PREFIX = os.path.join(MODELS_DIR, 'stage2')
FEATURE_ENGINEERING_PATH = os.path.join(MODELS_DIR, 'feature_engineering.pkl')

# Plan types (based on data analysis)
PLAN_TYPES = [
    'Long Term Disability',
    'Voluntary Life & AD&D',
    'Short Term Disability',
    'Vision',
    'Dental',
    'Critical Illness',
    'Accident',
    'Dependent Life - Spouse',
    'Dependent Life - Children',
    'Long Term Care',
    'Identity Theft',
    'Critical Illness - Spouse',
    'Critical Illness - Children',
    'Telemedicine',
    'Hospital Indemnity',
    'Health Cost Sharing',
    'Excess DI'
]

# Coverage tiers (based on data analysis)
COVERAGE_TIERS = [
    'Employee Only',
    'Employee Plus Spouse',
    'Family'
]

# Feature columns
DEMOGRAPHIC_FEATURES = [
    'Sex',
    'Age as of Billing Start Date',
    'Pay Frequency',
    'Estimated Annual Gross Income',
    'State',
    'Employment Status',
    'Job Class'
]

# Model hyperparameters
STAGE1_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced'  # Handle class imbalance
}

STAGE2_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

# Data processing parameters
OUTLIER_PERCENTILE_LOW = 1
OUTLIER_PERCENTILE_HIGH = 99
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_FOR_TIER_MODEL = 20

# Recommendation parameters
CONFIDENCE_THRESHOLD = 0.30  # 30% confidence threshold (lowered for better recall)
TOP_N_RECOMMENDATIONS = 10
SHOW_ALL_PLAN_SCORES = True  # Show scores for all plans in recommendations

# Age brackets for feature engineering
AGE_BRACKETS = [0, 30, 40, 50, 60, 100]
AGE_LABELS = ['<30', '30-40', '40-50', '50-60', '60+']

# Income brackets for feature engineering
INCOME_BRACKETS = [0, 50000, 100000, 200000, 500000, float('inf')]
INCOME_LABELS = ['<50K', '50K-100K', '100K-200K', '200K-500K', '500K+']

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
