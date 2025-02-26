import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained scaler and model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("best_logistic_regression.pkl", "rb"))

st.write("""
# Lung Cancer Prediction 
This app predicts the **Possibility of Lung Cancer** based on user inputs!
""")

st.sidebar.header("User Input Features")

# Function to get manual user input
def user_input_features():
    GENDER = st.sidebar.selectbox("Gender", ("Male", "Female", "Others"))
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

# Display Image and Information (Only Before Prediction)
def display_initial_info():
    st.image("image.jpg", caption="Lung Cancer Overview")  # Update the image path
    st.subheader("How to Prevent Lung Cancer?")
    st.write("""
    - **Quit Smoking:** Smoking is the leading cause of lung cancer.
    - **Limit Exposure to Toxins:** Avoid exposure to harmful substances.
    - **Regular Check-ups:** Early detection saves lives.
    """)
    st.subheader("Why Lung Cancer Happens?")
    st.write("""
    - **Smoking:** The major risk factor for lung cancer.
    - **Genetics:** Family history plays a role.
    - **Radon Exposure:** Inhalation of radon gas increases risk.
    """)
    st.subheader("Causes of Lung Cancer")
    st.write("""
    - **Cigarette Smoke:** Includes both active smoking and secondhand smoke.
    - **Air Pollution:** Long-term exposure to polluted air.
    - **Genetics:** Family history increases the chances.
    """)

# Load CSV File or Get User Inputs
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Show Example CSV Data
if st.sidebar.button("View Example Data"):
    example_data = pd.read_csv("Lung_cancer_detection.csv")  # Ensure correct file path
    st.subheader("Example Data")
    st.write(example_data.head(10))  # Show first 10 rows of the example CSV
    st.markdown("""
        **Legend:**
        - **1** → No
        - **2** → Yes
    """)

# Check if the user uploaded a file
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df.columns = input_df.columns.str.strip()
    st.sidebar.success("CSV uploaded successfully!")
else:
    input_df = user_input_features()

# If 'Predict' button is clicked
if st.sidebar.button("Predict"):
    # Scaling the input data
    data_scaled = scaler.transform(input_df)

    # Making predictions
    predictions = model.predict(data_scaled)
    predictions_proba = model.predict_proba(data_scaled)

    # Display prediction result and graph
    st.subheader("User Input Features")
    st.write(input_df)

    st.subheader("Predictions")
    input_df["Lung Cancer Risk"] = ["Yes" if pred == 1 else "No" for pred in predictions]
    st.write(input_df[["Lung Cancer Risk"]])

    st.subheader("Prediction Probability")
    probability_df = pd.DataFrame({
        "Lung Cancer": ["No", "Yes"],
        "Probability": [f"{predictions_proba[0][0]:.2f}", f"{predictions_proba[0][1]:.2f}"]
    })
    st.table(probability_df)

    
    fig, ax = plt.subplots()

    # Set colors based on which probability is higher
    colors = ['green' if predictions_proba[0][0] > predictions_proba[0][1] else 'grey', 
              'red' if predictions_proba[0][1] > predictions_proba[0][0] else 'grey']

    # Create a bar plot with custom colors
    sns.barplot(x=["No", "Yes"], y=predictions_proba[0], ax=ax, hue=["No", "Yes"], dodge=False, palette=colors)

    ax.set_title("Prediction Probability")
    ax.set_ylabel("Probability")

    # Display the plot
    st.pyplot(fig)



else:
    # Display initial image and info until "Predict" is clicked
    display_initial_info()
