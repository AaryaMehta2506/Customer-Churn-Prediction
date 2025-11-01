import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#Streamlit Page Setup
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.markdown("<h2 style='text-align:center; color:#3A86FF;'>Customer Churn Prediction Dashboard</h2>", unsafe_allow_html=True)
st.write("Predict whether a customer is likely to churn based on their details below")

#Load Model & Columns
try:
    model = joblib.load('churn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    # st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

#User Input Section
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

with col2:
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

#Create Input DataFrame 
input_dict = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_dict])

#One-hot encode
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

#Prediction Section 
st.header("Prediction Result")

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]  # probability of churn

        if prediction == 1:
            st.error(f"This customer is **likely to CHURN**.\n\nProbability: **{probability:.2%}**")
        else:
            st.success(f"This customer is **likely to STAY**.\n\nProbability: **{probability:.2%}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

#Feature Importance Section 
st.markdown("---")
st.header("Feature Importance (Top 10)")

if st.checkbox("Show Feature Importance"):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # top 10 important features
            plt.figure(figsize=(8, 5))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), np.array(model_columns)[indices])
            plt.xlabel("Importance")
            plt.title("Top 10 Features Influencing Churn")
            st.pyplot(plt)
        else:
            st.warning("Feature importance not available for this model type.")
    except Exception as e:
        st.error(f"Error displaying feature importance: {e}")

st.markdown("<br><hr><p style='text-align:center; color:grey;'>Developed by Aarya</p>", unsafe_allow_html=True)
