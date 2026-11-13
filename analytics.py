import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to 'AI Privacy proj' and then into 'data'
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "social_media_usage.csv")

try:
    df = pd.read_csv(CSV_PATH)
    print("CSV loaded successfully!")
except FileNotFoundError:
    print(f"Still cannot find CSV at: {os.path.abspath(CSV_PATH)}")

def usage_summary():
    if df is None:
        return {"error": "Data not loaded"}

    return {
        "total_users": int(len(df)),
        "average_posts_per_day": round(df["Posts_Per_Day"].mean(), 2),
        "average_likes_per_day": round(df["Likes_Per_Day"].mean(), 2),
        "average_follows_per_day": round(df["Follows_Per_Day"].mean(), 2)
    }
