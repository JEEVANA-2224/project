import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🩸",
    layout="centered"
)

st.title("🩸 Disease Prediction System")
st.write("Predict disease using ML models")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    lr = pickle.load(open("logistic_model.pkl", "rb"))
    knn = pickle.load(open("knn_model.pkl", "rb"))
    rf = pickle.load(open("rf_model.pkl", "rb"))
    bag = pickle.load(open("bagging_model.pkl", "rb"))
    ada = pickle.load(open("adaboost_model.pkl", "rb"))
    return scaler, lr, knn, rf, bag, ada

scaler, lr, knn, rf, bag, ada = load_models()

# ---------------- User Input ----------------
st.subheader("Enter Blood Test Values")

features = []
feature_names = [
    "Glucose",
    "Cholesterol",
    "Hemoglobin",
    "Platelets",
    "White Blood Cells",
    "Red Blood Cells",
    "Hematocrit",
    "Mean Corpuscular Volume",
    "Mean Corpuscular Hemoglobin",
    "Mean Corpuscular Hemoglobin Concentration",
    "Insulin",
    "BMI",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Triglycerides",
    "HbA1c",
    "LDL Cholesterol",
    "HDL Cholesterol",
    "ALT",
    "AST",
    "Heart Rate",
    "Creatinine",
    "Troponin",
    "C-reactive Protein"
]


for name in feature_names:
    val = st.number_input(name, value=0.0)
    features.append(val)

input_data = np.array(features).reshape(1, -1)

# ---------------- Model Selection ----------------
model_name = st.selectbox(
    "Choose Model",
    ("Logistic Regression", "KNN", "Random Forest", "Bagging", "AdaBoost")
)

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)

    if model_name == "Logistic Regression":
        prediction = lr.predict(input_scaled)
    elif model_name == "KNN":
        prediction = knn.predict(input_scaled)
    elif model_name == "Random Forest":
        prediction = rf.predict(input_scaled)
    elif model_name == "Bagging":
        prediction = bag.predict(input_scaled)
    else:
        prediction = ada.predict(input_scaled)

    st.success(f"🧬 Predicted Disease Class: {prediction[0]}")
