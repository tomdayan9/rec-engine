# Insurance Recommendation Engine - Complete Guide

A machine learning system that provides personalized insurance portfolio recommendations using a two-stage Random Forest approach. The system predicts which insurance plans a customer should have and the appropriate coverage tier for each plan.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train both Stage 1 and Stage 2 models (required before first use)
python main.py train
```

### 3. Get Recommendations

**Option A: Command Line Interface**
```bash
python main.py predict --age 35 --income 75000 --sex Male --state CA --job Professional
```

**Option B: Web Interface**
```bash
# Start the web server
python app.py

# Open your browser to: http://localhost:5000
```

**Option C: Demo Mode**
```bash
# See example recommendations for 3 sample customers
python main.py demo
```

## How It Works

### Architecture Overview

The system uses a **two-stage machine learning architecture**:

**Stage 1: Plan Type Prediction**
- Determines WHICH insurance plans the customer should have
- Multi-label classification (17 binary classifiers)
- Outputs a score (0-100%) for each plan type

**Stage 2: Coverage Tier Prediction**
- Determines the appropriate coverage level for each recommended plan
- Separate classifier for each plan type
- Outputs: Employee Only, Employee + Spouse, or Family

### Training Pipeline

```
Source Data (source.csv)
    ↓
Data Cleaning & Preprocessing
    ↓
Feature Engineering (47 features)
    ↓
Train/Test Split (80/20)
    ↓
Stage 1: Multi-Label Random Forest
    ↓
Stage 2: Individual Random Forests per Plan
    ↓
Save Models to models/
```

### Prediction Pipeline

```
Customer Demographics Input
    ↓
Feature Engineering
    ↓
Stage 1: Predict Plan Scores
    ↓
Stage 2: Predict Coverage Tiers
    ↓
Premium Estimation
    ↓
Comprehensive Recommendation Output
```

## Project Structure

```
insurance-recommendation-engine/
├── source.csv                  # Raw data (customer insurance records)
├── config.py                   # Configuration & hyperparameters
├── main.py                     # CLI interface
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── cakewalk_design.md         # Design system specification
│
├── data/                       # Processed data (generated)
│   ├── processed_data.pkl
│   ├── train_data.pkl
│   └── test_data.pkl
│
├── models/                     # Trained models (generated)
│   ├── stage1_plan_predictor.pkl
│   ├── stage2_tier_predictors.pkl
│   └── feature_engineering.pkl
│
├── src/                        # Source code
│   ├── data_processing.py      # Data cleaning & restructuring
│   ├── feature_engineering.py  # Feature extraction & encoding
│   ├── stage1_model.py         # Plan type predictor
│   ├── stage2_model.py         # Coverage tier predictors
│   ├── recommendation.py       # Original recommendation engine
│   ├── recommendation_v2.py    # Improved engine (used by default)
│   ├── evaluation.py           # Model evaluation
│   └── utils.py                # Helper functions
│
├── templates/                  # Web UI templates
│   └── index.html
│
├── static/                     # Web UI assets
│   └── js/
│       └── app.js
│
└── tests/                      # Unit tests
    └── test_recommendation.py
```

## Command Reference

### Training

```bash
# Train all models (must run first before predictions)
python main.py train
```

**What it does:**
- Loads and cleans source.csv
- Removes outliers and missing data
- Creates 47 engineered features
- Trains Stage 1 multi-label classifier
- Trains Stage 2 tier prediction models (5 models)
- Saves all models to models/ directory

**Expected time:** 30-60 seconds

### Evaluation

```bash
# Evaluate trained models on test data
python main.py evaluate
```

**What it does:**
- Loads trained models
- Evaluates on test set (20% of data)
- Shows per-plan accuracy, precision, recall, F1
- Shows feature importance rankings
- Generates evaluation_report.txt

### Prediction

```bash
python main.py predict \
  --age <age> \
  --income <annual_income> \
  --sex <Male|Female> \
  --state <state_code> \
  --job <job_class> \
  [--confidence <0-1>]
```

**Required Parameters:**
- `--age`: Customer age (e.g., 35)
- `--income`: Annual gross income (e.g., 75000)
- `--sex`: Male or Female
- `--state`: State abbreviation (e.g., CA, NY, TX)
- `--job`: Professional, Agent, Management, or Staff

**Optional Parameters:**
- `--confidence`: Threshold for recommendations (default: 0.30)
- `--employment-status`: Employment status (default: A for Active)
- `--pay-frequency`: Pay frequency (default: 12 for monthly)

**Examples:**
```bash
# Young professional in California
python main.py predict --age 28 --income 55000 --sex Male --state CA --job Professional

# Mid-career agent in New York with family
python main.py predict --age 45 --income 150000 --sex Female --state NY --job Agent

# Senior executive in Texas
python main.py predict --age 55 --income 300000 --sex Male --state TX --job Agent

# Custom confidence threshold (show more recommendations)
python main.py predict --age 35 --income 75000 --sex Male --state CA --confidence 0.2
```

### Demo Mode

```bash
# Run 3 example scenarios
python main.py demo
```

Shows recommendations for:
1. Young Professional (28, $55K, CA)
2. Mid-Career Agent (45, $150K, NY)
3. Senior Executive (55, $300K, TX)

## Web Interface

### Starting the Server

```bash
python app.py
```

The server starts at: **http://localhost:5000**

### Features

- Clean, user-friendly form for entering customer demographics
- Comprehensive recommendation display with scores
- Color-coded recommendation levels (Highly Recommended, Recommended, Consider, Not Recommended)
- Premium estimates for all plans
- Total monthly and annual premium calculation
- Responsive design (mobile-friendly)
- Follows Cakewalk Benefits Platform design system

### API Endpoint

**POST /api/recommend**

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "sex": "Male",
    "state": "CA",
    "job_class": "Professional"
  }'
```

**Response:**
```json
{
  "customer": {...},
  "plans": [
    {
      "name": "Long Term Disability",
      "score": 78.5,
      "recommendation_level": "Highly Recommended",
      "coverage_tier": "Employee Only",
      "premium": 93.76,
      "recommended": true
    },
    ...
  ],
  "summary": {
    "total_recommended": 9,
    "total_monthly_premium": 214.57,
    "total_annual_premium": 2574.84
  }
}
```

## Understanding the Output

### Recommendation Format

```
PLAN TYPE                           SCORE      RECOMMENDATION         COVERAGE TIER              PREMIUM
----------------------------------------------------------------------------------------------------
✓ Long Term Disability              78.5%      Highly Recommended     Employee Only              $93.76/mo
✓ Short Term Disability             65.2%      Recommended            Employee Only              $45.30/mo
✓ Dental                           52.3%      Recommended            Employee Plus Spouse       $62.15/mo
○ Vision                           45.1%      Consider               Employee Only              $12.40/mo
✗ Critical Illness                 28.5%      Not Recommended        Employee Only              $35.20/mo
```

**Symbols:**
- `✓` = Recommended (score >= threshold)
- `✗` = Not Recommended (score < threshold)

**Recommendation Levels:**
- **Highly Recommended:** Score >= 70%
- **Recommended:** Score >= 50%
- **Consider:** Score >= threshold (default 30%)
- **Not Recommended:** Score < threshold

### Confidence Threshold

The confidence threshold (default 30%) determines which plans appear as "recommended" (✓).

**Adjusting the threshold:**
- Lower threshold (0.2) = More recommendations (higher recall)
- Higher threshold (0.5) = Fewer recommendations (higher precision)

## Configuration

Edit [config.py](config.py) to customize:

### Model Hyperparameters

```python
STAGE1_PARAMS = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 10,            # Maximum tree depth
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced'  # Handle class imbalance
}

STAGE2_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}
```

### Data Processing

```python
TEST_SIZE = 0.2                      # 20% test set
CONFIDENCE_THRESHOLD = 0.30          # 30% minimum score
MIN_SAMPLES_FOR_TIER_MODEL = 20     # Minimum samples for tier model
OUTLIER_PERCENTILE_LOW = 1          # Remove bottom 1%
OUTLIER_PERCENTILE_HIGH = 99        # Remove top 1%
```

### Feature Engineering

```python
AGE_BRACKETS = [0, 30, 40, 50, 60, 100]
INCOME_BRACKETS = [0, 50000, 100000, 200000, 500000, float('inf')]
```

## Model Performance

**Stage 1: Plan Type Prediction**
- Average accuracy: 85.77%
- Hamming loss: 0.1423
- Best predictions: Long Term Disability (89.58% F1), Short Term Disability (64.04% F1)

**Stage 2: Coverage Tier Prediction**
- Average accuracy: 74.77%
- Best predictions: Accident (82.14%), Critical Illness (79.69%), Dental (73.91%)

**Feature Importance (Top 5):**
1. Estimated Annual Gross Income (19.1%)
2. Age as of Billing Start Date (17.1%)
3. Pay Frequency (4.0%)
4. Job Class: Management (3.7%)
5. Job Class: Agent (3.1%)

## Data Requirements

The source data file ([source.csv](source.csv)) must contain:

**Required Columns:**
- Sex
- Age as of Billing Start Date
- Pay Frequency
- Estimated Annual Gross Income
- State
- ZIP Code
- Employment Status
- Job Class
- Plan Type
- Coverage Tier
- Coverage Amount
- Monthly Premium
- Coverage Effective Date

**Format:** One row per insurance plan enrollment (multiple rows per customer)

**Data Statistics:**
- 1,442 unique customers
- 4,621 total plan enrollments
- 17 different plan types
- 3 coverage tiers

## Insurance Plans Supported

The system can recommend from 17 plan types:

1. Long Term Disability
2. Short Term Disability
3. Voluntary Life & AD&D
4. Dental
5. Vision
6. Critical Illness
7. Accident
8. Dependent Life - Spouse
9. Dependent Life - Children
10. Long Term Care
11. Identity Theft
12. Critical Illness - Spouse
13. Critical Illness - Children
14. Telemedicine
15. Hospital Indemnity
16. Health Cost Sharing
17. Excess DI

**Coverage Tiers:**
- Employee Only
- Employee Plus Spouse
- Family

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Or:

```bash
python tests/test_recommendation.py
```

## Troubleshooting

**Error: "Models not trained yet"**
```bash
# Solution: Train models first
python main.py train
```

**Error: "File not found: source.csv"**
```bash
# Solution: Ensure source.csv is in the project root
ls source.csv
```

**Error: "ModuleNotFoundError"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Low model performance after training:**
- Check data quality in source.csv
- Adjust hyperparameters in config.py
- Increase training data size

**Web server won't start (port in use):**
```bash
# Kill existing Flask process
pkill -f "python app.py"

# Or use a different port in app.py
# app.run(debug=True, port=5001)
```

**Web server: Models not loading:**
```bash
# Ensure models are trained
python main.py train

# Check models directory exists
ls -la models/
```

## Development

### Adding New Features

**To add a new demographic feature:**
1. Add to `DEMOGRAPHIC_FEATURES` in config.py
2. Update feature engineering in [src/feature_engineering.py](src/feature_engineering.py)
3. Retrain models: `python main.py train`

**To modify algorithms:**
1. Edit [src/stage1_model.py](src/stage1_model.py) or [src/stage2_model.py](src/stage2_model.py)
2. Update hyperparameters in config.py
3. Retrain: `python main.py train`

**To add new plan types:**
1. Add to `PLAN_TYPES` in config.py
2. Ensure plan exists in source.csv
3. Retrain: `python main.py train`

### Code Structure

**Core Components:**
- [config.py](config.py): All configuration
- [main.py](main.py): CLI interface
- [app.py](app.py): Web server

**Data Processing:**
- [src/data_processing.py](src/data_processing.py): Data cleaning, restructuring, train/test split
- [src/feature_engineering.py](src/feature_engineering.py): Feature extraction, encoding, scaling

**Models:**
- [src/stage1_model.py](src/stage1_model.py): Multi-label RandomForest for plan prediction
- [src/stage2_model.py](src/stage2_model.py): Individual RandomForests for tier prediction

**Recommendation:**
- [src/recommendation_v2.py](src/recommendation_v2.py): Improved engine with score-based output (used by default)
- [src/recommendation.py](src/recommendation.py): Original engine

**Evaluation:**
- [src/evaluation.py](src/evaluation.py): Model evaluation and metrics

**Utilities:**
- [src/utils.py](src/utils.py): Helper functions

## Dependencies

Key libraries:
- **scikit-learn**: Machine learning models
- **pandas**: Data processing
- **numpy**: Numerical operations
- **Flask**: Web server
- **joblib**: Model serialization

See [requirements.txt](requirements.txt) for complete list.

## Design System

The web interface follows the **Cakewalk Benefits Platform Design System** specified in [cakewalk_design.md](cakewalk_design.md):

- Typography: Space Grotesk & DM Sans
- Colors: Primary blue (#005dfe), success green (#15cb94)
- Components: Rounded cards, proper spacing (8pt grid)
- Accessibility: WCAG compliant, keyboard navigation
- Tracking: Data attributes on all interactive elements

## Production Deployment

For production use:

1. **Use a production WSGI server:**
```bash
# Option 1: Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Option 2: Waitress
pip install waitress
waitress-serve --port=5000 app:app
```

2. **Set environment variables:**
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
```

3. **Security considerations:**
- Use HTTPS
- Add rate limiting
- Implement authentication if needed
- Add input validation and sanitization
- Set proper CORS policies
- Use environment variables for sensitive config

## Support

For issues:
1. Check this README
2. Run `python main.py evaluate` to check model performance
3. Check the design system in [cakewalk_design.md](cakewalk_design.md)
4. Review source code documentation

## License

This project is provided as-is for educational and commercial use.
