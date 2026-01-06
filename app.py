import streamlit as st
import pickle
import pandas as pd

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

# 🔑 This is the KEY FIX
expected_columns = scaler.feature_names_in_

# create full input with ALL 24 features
input_data = pd.DataFrame(columns=expected_columns)
input_data.loc[0] = 0.0   # default values

# collect user inputs (only 19, rest auto-filled)
user_inputs = {
    "Glucose": st.number_input("Glucose", 0.0),
    "Cholesterol": st.number_input("Cholesterol", 0.0),
    "Hemoglobin": st.number_input("Hemoglobin", 0.0),
    "Platelets": st.number_input("Platelets", 0.0),
    "White Blood Cells": st.number_input("White Blood Cells", 0.0),
    "Red Blood Cells": st.number_input("Red Blood Cells", 0.0),
    "Hematocrit": st.number_input("Hematocrit", 0.0),
    "Mean Corpuscular Volume": st.number_input("Mean Corpuscular Volume", 0.0),
    "Mean Corpuscular Hemoglobin": st.number_input("Mean Corpuscular Hemoglobin", 0.0),
    "Mean Corpuscular Hemoglobin Concentration": st.number_input(
        "Mean Corpuscular Hemoglobin Concentration", 0.0
    ),
    "HbA1c": st.number_input("HbA1c", 0.0),
    "LDL Cholesterol": st.number_input("LDL Cholesterol", 0.0),
    "HDL Cholesterol": st.number_input("HDL Cholesterol", 0.0),
    "ALT": st.number_input("ALT", 0.0),
    "AST": st.number_input("AST", 0.0),
    "Heart Rate": st.number_input("Heart Rate", 0.0),
    "Creatinine": st.number_input("Creatinine", 0.0),
    "Troponin": st.number_input("Troponin", 0.0),
    "C-reactive Protein": st.number_input("C-reactive Protein", 0.0),
}

# assign inputs to dataframe
for col, val in user_inputs.items():
    input_data[col] = val

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
