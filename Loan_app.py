import streamlit as st
import joblib
import pandas as pd

model = joblib.load("loan_model.pkl")
scaler = joblib.load("loan_scaler.pkl")

st.title("🏦 Loan Eligibility Predictor")

income = st.number_input("Applicant Income", 0, 100000)
co_income = st.number_input("Coapplicant Income", 0, 100000)
loan_amt = st.number_input("Loan Amount", 0, 1000)
term = st.selectbox("Loan Term (Months)", [360, 180, 120, 60])
credit = st.selectbox("Credit History", [0, 1])
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding
input_data = pd.DataFrame({
    "Gender": [1 if gender == "Male" else 0],
    "Married": [1 if married == "Yes" else 0],
    "Dependents": [dependents],
    "Education": [0 if education == "Graduate" else 1],
    "Self_Employed": [1 if self_employed == "Yes" else 0],
    "ApplicantIncome": scaler.transform([[income, co_income, loan_amt]])[0][0],
    "CoapplicantIncome": scaler.transform([[income, co_income, loan_amt]])[0][1],
    "LoanAmount": scaler.transform([[income, co_income, loan_amt]])[0][2],
    "Loan_Amount_Term": [term],
    "Credit_History": [credit],
    "Property_Area": [property_area.index(property_area)]
})

if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

   
