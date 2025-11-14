"""
Flask web application for Insurance Recommendation Engine.
Provides a simple web UI for getting insurance recommendations.
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.recommendation_v2 import ImprovedRecommendationEngine

app = Flask(__name__)

# Initialize the recommendation engine
engine = ImprovedRecommendationEngine()
engine.load_models()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for getting insurance recommendations.

    Expected JSON:
    {
        "age": 35,
        "income": 75000,
        "sex": "Male",
        "state": "CA",
        "job_class": "Professional"
    }
    """
    try:
        data = request.json

        # Validate required fields
        required_fields = ['age', 'income', 'sex', 'state', 'job_class']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create customer data
        customer_data = {
            'Sex': data['sex'],
            'Age as of Billing Start Date': int(data['age']),
            'Pay Frequency': 12,  # Monthly
            'Estimated Annual Gross Income': float(data['income']),
            'State': data['state'],
            'Employment Status': 'A',  # Active
            'Job Class': data['job_class']
        }

        # Get recommendations
        recommendations = engine.recommend_comprehensive(customer_data)

        # Format response
        response = {
            'customer': {
                'age': customer_data['Age as of Billing Start Date'],
                'income': customer_data['Estimated Annual Gross Income'],
                'sex': customer_data['Sex'],
                'state': customer_data['State'],
                'job_class': customer_data['Job Class']
            },
            'plans': [],
            'summary': {
                'total_recommended': recommendations['total_plans_recommended'],
                'total_monthly_premium': recommendations['total_monthly_premium'],
                'total_annual_premium': recommendations['total_annual_premium']
            }
        }

        # Format each plan
        for plan in recommendations['all_plans']:
            # Handle premium - check if it's a valid number
            premium = plan['estimated_premium']
            if pd.isna(premium) or np.isnan(premium):
                premium = None

            response['plans'].append({
                'name': plan['plan_type'],
                'score': round(plan['score'] * 100, 1),
                'recommendation_level': plan['recommendation_level'],
                'coverage_tier': plan['coverage_tier'],
                'premium': premium,
                'recommended': bool(plan['recommended'])  # Convert numpy bool to Python bool
            })

        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/recommend: {error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500


if __name__ == '__main__':
    import pandas as pd
    app.run(debug=False, port=5000)
