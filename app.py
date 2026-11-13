import streamlit as st
import requests
import pandas as pd

# Set Page Config
st.set_page_config(page_title="Social Media Usage Predictor", layout="wide")

st.title("📱 Social Media Behavior Analysis")

# --- STEP 1: CREATE TABS ---
tab1, tab2 = st.tabs(["🔍 Predict User Type", "📜 Prediction History"])

# --- TAB 1: PREDICTION LOGIC ---
with tab1:
    st.header("Predict Behavior")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            minutes = st.number_input("Daily Minutes Spent", min_value=0.0, value=120.0)
            posts = st.number_input("Posts Per Day", min_value=0, value=5)
        with col2:
            likes = st.number_input("Likes Per Day", min_value=0, value=50)
            follows = st.number_input("Follows Per Day", min_value=0, value=10)
        
        submit = st.form_submit_button("Run Analysis")

    # In your frontend code, update the response parsing:
if submit:
    payload = {
        "Daily_Minutes_Spent": minutes,
        "Posts_Per_Day": posts,
        "Likes_Per_Day": likes,
        "Follows_Per_Day": follows
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if there's an error from backend
            if "error" in result:
                st.error(f"Backend Error: {result['error']}")
            else:
                # Now using the correct keys
                prediction = result.get("prediction")
                confidence = result.get("confidence")
                
                st.divider()
                if prediction == 1:
                    st.error(f"Outcome: **HEAVY USER**")
                else:
                    st.success(f"Outcome: **NORMAL USER**")
                st.info(f"AI Confidence Score: {confidence}")
        else:
            st.error(f"Backend error (Status: {response.status_code})")
            
    except requests.exceptions.ConnectionError:
        st.error("Error: Could not connect to backend. Ensure main.py is running on port 8000.")

# --- TAB 2: HISTORY LOGIC (PHASE 2) ---
with tab2:
    st.header("Local Prediction Logs")
    st.write("This data is pulled directly from the local JSON storage (Privacy-First).")

    if st.button("Refresh History"):
        try:
            # Call Backend History Endpoint
            response = requests.get("http://127.0.0.1:8000/history")
            history_data = response.json()

            if history_data:
                # Convert JSON list to a Pandas DataFrame for a pretty table
                df_history = pd.DataFrame(history_data)
                
                # Reorder columns to show Timestamp first
                cols = ['timestamp', 'prediction', 'confidence', 'Daily_Minutes_Spent', 'Posts_Per_Day', 'Likes_Per_Day', 'Follows_Per_Day']
                df_history = df_history[cols]

                # Display the table
                st.dataframe(df_history, use_container_width=True)
                
                # Optional: Add a simple metric count
                st.metric("Total Predictions Stored", len(df_history))
            else:
                st.warning("No history found yet. Make a prediction first!")

        except Exception as e:
            st.error("Could not fetch history from backend.")