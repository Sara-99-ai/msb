# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import pandas as pd

# Load the model (check if it's a dictionary)
pipeline = joblib.load('logistic_linear_models.pkl')

# Check if the loaded object is a dictionary
if isinstance(pipeline, dict):
    # Access the classification and regression models from the dictionary
    classifier = pipeline.get('classification_model')  # Assuming the key is 'classification_model'
    regressor = pipeline.get('regression_model')       # Assuming the key is 'regression_model'
else:
    st.error("The loaded pipeline is not a dictionary. Please check the model file.")
    st.stop()

# Streamlit page configuration
st.set_page_config(page_title="Mortgage Prediction Application", layout="wide")

# Custom CSS for styling (omitted for simplicity in this example)

st.title(":blue[Mortgage Prediction Application]")

# Collect user inputs
st.header(":green[Enter Details]")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=0, step=1)
    dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0, value=30, step=1)
    orig_upb = st.number_input("Original UPB (Unpaid Principal Balance)", min_value=0, value=150000, step=1000)
    ltv = st.number_input("Loan-to-Value Ratio (%)", min_value=0, value=80, step=1)
    orig_interest_rate = st.number_input("Original Interest Rate (%)", min_value=0.0, value=3.5, step=0.01)
    ever_delinquent = st.selectbox("Ever Delinquent (0 or 1)", [0, 1])
    months_delinquent = st.number_input("Months Delinquent", min_value=0, value=0, step=1)
    months_in_repayment = st.number_input("Months in Repayment", min_value=0, value=24, step=1)

with col2:
    loan_age_years = st.number_input("Loan Age (years)", min_value=0, value=2, step=1)
    monthly_payment = st.number_input("Monthly Payment", min_value=0, value=1000, step=50)
    total_payment = st.number_input("Total Payment", min_value=0, value=24000, step=100)
    interest_amount = st.number_input("Interest Amount", min_value=0, value=5000, step=100)
    cur_principal = st.number_input("Current Principal", min_value=0, value=100000, step=1000)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=5000, step=100)
    prepayment = st.number_input("Prepayment", min_value=0, value=0, step=100)

# Create a DataFrame from user inputs
user_input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'DTI': dti,
    'OrigUPB': orig_upb,
    'LTV': ltv,
    'OrigInterestRate': orig_interest_rate,
    'EverDelinquent': ever_delinquent,
    'MonthsDelinquent': months_delinquent,
    'MonthsInRepayment': months_in_repayment,
    'LoanAge_years': loan_age_years,
    'monthly_payment': monthly_payment,
    'total_payment': total_payment,
    'interest_amount': interest_amount,
    'cur_principal': cur_principal,
    'monthly_income': monthly_income,
    'prepayment': prepayment
}])

# Display the DataFrame to the user
st.write("### :orange[User Input DataFrame:]")
st.write(user_input_df)

# Predict and display results
if st.button('Predict Classification and Regression'):
    try:
        # Get classification and regression predictions
        y_class_pred = classifier.predict(user_input_df)
        y_reg_pred = regressor.predict(user_input_df)

        # Display classification result
        st.subheader("Prediction Results")
        st.write(f"*Classification Prediction (Ever Delinquent):* {y_class_pred[0]}")

        # Display regression result
        st.write(f"*Regression Prediction (Prepayment):* {y_reg_pred[0]}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
