from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Add this
from model import predict_heavy_user
from analytics import usage_summary
import pandas as pd
import json
import joblib 
import os 
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(
    title="Social Media Usage Backend",
    description="Phase-1 Engagement-Based Heavy User Classification",
    version="1.0"
)

# 🔥 CRITICAL: Add CORS middleware to allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserData(BaseModel):
    Daily_Minutes_Spent: float
    Posts_Per_Day: int
    Likes_Per_Day: int
    Follows_Per_Day: int

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_heavy_user_model1.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
else:
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
    model = None

def save_prediction_to_json(user_data: UserData, prediction: int, confidence: float):
    """Save prediction to JSON file"""
    # 1. Prepare the new entry
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Daily_Minutes_Spent": user_data.Daily_Minutes_Spent,
        "Posts_Per_Day": user_data.Posts_Per_Day,
        "Likes_Per_Day": user_data.Likes_Per_Day,
        "Follows_Per_Day": user_data.Follows_Per_Day,
        "prediction": int(prediction),  # Changed to match frontend
        "confidence": round(float(confidence), 2)
    }

    # 2. Read existing data or create new file
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except (json.JSONDecodeError, FileNotFoundError):
            history = []
    else:
        history = []
        print(f"📁 Creating new history file at {HISTORY_FILE}")

    # 3. Append and Save
    history.append(new_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"✅ Prediction saved to history (Total entries: {len(history)})")

@app.get("/")
def root():
    return {
        "message": "Social Media Usage API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "history": "/history",
            "summary": "/summary"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "Backend is running"}

@app.get("/summary")
def summary():
    return usage_summary()

@app.post("/predict")
def predict_user(data: UserData):
    if model is None:
        return {"error": "Model not loaded", "prediction": 0, "confidence": 0}
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Save to history
        save_prediction_to_json(data, prediction, probability)
        
        # 🔥 RETURN KEYS THAT MATCH FRONTEND EXPECTATIONS
        return {
            "prediction": int(prediction),  # Changed from "is_heavy_user"
            "confidence": round(float(probability), 2)  # Changed from "confidence_score"
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {"error": str(e), "prediction": 0, "confidence": 0}

@app.get("/history")
def get_history():
    """Return prediction history"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
                print(f"📋 Returning {len(history)} history entries")
                return history
        except json.JSONDecodeError:
            print("⚠️ History file corrupted, returning empty list")
            return []
    else:
        print("📁 No history file found")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)