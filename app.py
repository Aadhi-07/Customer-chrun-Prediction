from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="SaaS User Retention API")

# Load model and encoders
MODEL_PATH = "churn_pipeline.pkl"
ENCODER_PATH = "encoders.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
else:
    model = None
    encoders = None

# Define the input schema for validation
class UserData(BaseModel):
    gender: str
    age: int
    country: str
    city: str
    customer_segment: str
    tenure_months: int
    signup_channel: str
    contract_type: str
    monthly_logins: int
    weekly_active_days: int
    avg_session_time: float
    features_used: int
    usage_growth_rate: float
    last_login_days_ago: int
    monthly_fee: float
    total_revenue: float
    payment_method: str
    payment_failures: int
    discount_applied: str
    price_increase_last_3m: str
    support_tickets: int
    avg_resolution_time: float
    complaint_type: str
    csat_score: float
    escalations: int
    email_open_rate: float
    marketing_click_rate: float
    nps_score: int
    survey_response: str
    referral_count: int

@app.get("/")
def home():
    return {"message": "SaaS Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(user: UserData):
    if model is None or encoders is None:
        raise HTTPException(status_code=500, detail="Model or Encoders not loaded")

    try:
        # Convert Pydantic model to dict
        input_dict = user.dict()
        
        # Apply the saved LabelEncoders to categorical fields
        for col, le in encoders.items():
            val = input_dict.get(col)
            # Handle unseen labels by defaulting to the first class if necessary
            if val in le.classes_:
                input_dict[col] = int(le.transform([val])[0])
            else:
                input_dict[col] = 0 

        # Convert to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "status": "success",
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 3)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 