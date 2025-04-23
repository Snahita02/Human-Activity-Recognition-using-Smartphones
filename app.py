
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("har_model.pkl")

# Labels mapping
activity_labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

st.title("🏃 Human Activity Recognition App")
st.write("Upload a file with sensor data to predict activity:")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        prediction = model.predict(input_data)
        activities = [activity_labels[p] for p in prediction]
        
        st.success("Predicted Activities:")
        for i, act in enumerate(activities):
            st.write(f"Row {i+1}: {act}")
    except Exception as e:
        st.error(f"Error: {e}")
