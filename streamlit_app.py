import streamlit as st
import pandas as pd
import numpy as np
import pickle

scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("best_logistic_regression.pkl", "rb"))


st.write("""
# Lung Cancer Prediction App
This app predicts the **Possibility of Lung Cancer** based on user inputs!
""")

st.sidebar.header("User Input Features")


def user_input_features():
    GENDER = st.sidebar.selectbox("Gender", ("Male", "Female"))
    AGE = st.sidebar.slider("Age", 20, 90, 50)
    
    options = {"No": 1, "Yes": 2}  
    
    SMOKING = options[st.sidebar.selectbox("Smoking", ("No", "Yes"))]
    YELLOW_FINGERS = options[st.sidebar.selectbox("Yellow Fingers", ("No", "Yes"))]
    ANXIETY = options[st.sidebar.selectbox("Anxiety", ("No", "Yes"))]
    PEER_PRESSURE = options[st.sidebar.selectbox("Peer Pressure", ("No", "Yes"))]
    CHRONIC_DISEASE = options[st.sidebar.selectbox("Chronic Disease", ("No", "Yes"))]
    FATIGUE = options[st.sidebar.selectbox("Fatigue", ("No", "Yes"))]
    ALLERGY = options[st.sidebar.selectbox("Allergy", ("No", "Yes"))]
    WHEEZING = options[st.sidebar.selectbox("Wheezing", ("No", "Yes"))]
    ALCOHOL_CONSUMING = options[st.sidebar.selectbox("Alcohol Consuming", ("No", "Yes"))]
    COUGHING = options[st.sidebar.selectbox("Coughing", ("No", "Yes"))]
    SHORTNESS_OF_BREATH = options[st.sidebar.selectbox("Shortness of Breath", ("No", "Yes"))]
    SWALLOWING_DIFFICULTY = options[st.sidebar.selectbox("Swallowing Difficulty", ("No", "Yes"))]
    CHEST_PAIN = options[st.sidebar.selectbox("Chest Pain", ("No", "Yes"))]

    data = {
        "GENDER": 1 if GENDER == "Male" else 2, 
        "AGE": AGE,
        "SMOKING": SMOKING,
        "YELLOW_FINGERS": YELLOW_FINGERS,  
        "ANXIETY": ANXIETY,
        "PEER_PRESSURE": PEER_PRESSURE,  
        "CHRONIC DISEASE": CHRONIC_DISEASE,  
        "FATIGUE ": FATIGUE,  
        "ALLERGY ": ALLERGY,  
        "WHEEZING": WHEEZING,
        "ALCOHOL CONSUMING": ALCOHOL_CONSUMING,  
        "COUGHING": COUGHING,
        "SHORTNESS OF BREATH": SHORTNESS_OF_BREATH,  
        "SWALLOWING DIFFICULTY": SWALLOWING_DIFFICULTY,  
        "CHEST PAIN": CHEST_PAIN  
    }

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()


feature_names = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]


input_scaled = scaler.transform(pd.DataFrame(input_df, columns=feature_names))


prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)


st.subheader("User Input Features")
st.write(input_df)


st.subheader("Prediction")
st.write("Lung Cancer Risk: **Yes**" if prediction[0] == 1 else "Lung Cancer Risk: **No**")


st.subheader("Prediction Probability")
probability_df = pd.DataFrame({
    "Lung Cancer": ["No", "Yes"],
    "Probability": [f"{prediction_proba[0][0]:.2f}", f"{prediction_proba[0][1]:.2f}"]
})
st.table(probability_df)
