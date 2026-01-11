# SaaS Customer Churn Prediction

Predicting user churn using Machine Learning and serving via FastAPI.

## 📊 Model Details
- **Algorithm**: Logistic Regression
- **Performance**: 0.894 Accuracy
- **Features**: 30 behavioral and demographic variables

## 🚀 How to Run
1. Start the server: `python -m uvicorn app:app --reload`
2. Test the API: `python test_api.py`
3. View Docs: Visit `http://127.0.0.1:8000/docs` in your browser.

## 📁 Repository Structure
- `App.ipynb`: Data analysis and model training.
- `app.py`: FastAPI implementation.
- `churn_pipeline.pkl`: The trained model.
- `saas_users.csv`: User dataset.
