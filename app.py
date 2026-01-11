from fastapi import FastAPI
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI
app = FastAPI(title="SaaS User Retention API")

# 2. Check if the model exists before loading to prevent hanging
model_path = "churn_pipeline.pkl"
if os.path.exists(model_path):
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    print(f"ERROR: {model_path} not found.")
    model = None

@app.get("/")
def home():
    """Root endpoint to verify the API is alive."""
    return {"message": "SaaS Churn Prediction API is running"}
# Mappings based on LabelEncoder alphabetical ordering
# Note: Double-check these against your notebook's LabelEncoder.classes_
gender_map = {"Female": 0, "Male": 1}
country_map = {"Australia": 0, "Bangladesh": 1, "Canada": 2, "Germany": 3, "USA": 4}
contract_map = {"Annual": 0, "Monthly": 1, "Yearly": 2}

@app.post("/predict")
def predict_churn(data: dict):
    try:
        # Create a copy of input data to modify
        input_data = data.copy()

        # Manually encode categorical strings to numbers
        if "gender" in input_data:
            input_data["gender"] = gender_map.get(input_data["gender"], 0)
        
        if "country" in input_data:
            input_data["country"] = country_map.get(input_data["country"], 0)
            
        if "contract_type" in input_data:
            input_data["contract_type"] = contract_map.get(input_data["contract_type"], 0)

        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "status": "success",
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 3)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}