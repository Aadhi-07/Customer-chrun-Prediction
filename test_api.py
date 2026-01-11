import requests

url = "http://127.0.0.1:8000/predict" # Note: FastAPI default port is 8000
sample_user = {
    "gender": "Female", "age": 28, "country": "Canada", "city": "Toronto",
    "customer_segment": "Enterprise", "tenure_months": 24, "signup_channel": "Referral",
    "contract_type": "Annual", "monthly_logins": 25, "weekly_active_days": 6,
    "avg_session_time": 30.0, "features_used": 12, "usage_growth_rate": 0.15,
    "last_login_days_ago": 1, "monthly_fee": 120.0, "total_revenue": 2880.0,
    "payment_method": "PayPal", "payment_failures": 0, "discount_applied": "Yes",
    "price_increase_last_3m": "No", "support_tickets": 0, "avg_resolution_time": 0.0,
    "complaint_type": "None", "csat_score": 5.0, "escalations": 0,
    "email_open_rate": 0.8, "marketing_click_rate": 0.3, "nps_score": 9,
    "survey_response": "Very Satisfied", "referral_count": 5
}

response = requests.post(url, json=sample_user)
print(f"Status Code: {response.status_code}")
print(f"Prediction: {response.json()}") 