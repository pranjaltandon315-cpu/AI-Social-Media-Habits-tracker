import joblib
import pandas as pd
import os

# Resolve project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# Load trained model
MODEL_PATH = os.path.join(PROJECT_ROOT, "final_heavy_user_model1.pkl")
model = joblib.load(MODEL_PATH)

# Feature names used during training
FEATURE_NAMES = list(model.feature_names_in_)


def predict_heavy_user(data: dict):
    """
    Predicts whether a user is a Heavy User (engagement-based)
    """

    df = pd.DataFrame([data])

    if "App" in df.columns:
        df = pd.get_dummies(df, columns=["App"], drop_first=False)


    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

  
    df = df[FEATURE_NAMES]

    prediction = model.predict(df)[0]
    return int(prediction)
