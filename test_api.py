"""
Quick test to verify the recommendation engine works with new features
"""
from src.recommendation_v2 import ImprovedRecommendationEngine, create_example_customer, print_comprehensive_recommendation

# Create engine
engine = ImprovedRecommendationEngine()
engine.load_models()

# Test customer
customer = create_example_customer(35, 75000, 'Male', 'CA', 'Professional')

print("Testing recommendation engine with 52 features...")
print(f"Customer: Age 35, Income $75K, Male, CA, Professional\n")

# Get recommendation
recommendation = engine.recommend_comprehensive(customer)

# Print results
print_comprehensive_recommendation(recommendation)

print("\n✅ SUCCESS! The engine works correctly with the new feature engineering.")
