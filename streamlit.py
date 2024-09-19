import streamlit as st
import joblib
import pandas as pd

# Load the model dictionary
model_dict = joblib.load('logistic_linear_models.pkl')

# Access the logistic regression model (for classification)
logistic_model = model_dict.get('Logistic Regression')

# Access the linear regression model (for regression)
linear_model = model_dict.get('Linear Regression')

# Streamlit page configuration
st.set_page_config(page_title="Mortgage Prediction Application", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Mortgage Prediction Application")

# Collect user inputs
st.header("Enter Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=0, step=1)
    dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0, value=30, step=1)
    orig_upb = st.number_input("Original UPB", min_value=0, value=150000, step=1000)
    ltv = st.number_input("Loan-to-Value Ratio (%)", min_value=0, value=80, step=1)
    orig_interest_rate = st.number_input("Original Interest Rate (%)", min_value=0.0, value=3.5, step=0.01)
    ever_delinquent = st.selectbox("Ever Delinquent (0 or 1)", [0, 1])

with col2:
    months_delinquent = st.number_input("Months Delinquent", min_value=0, value=0, step=1)
    months_in_repayment = st.number_input("Months in Repayment", min_value=0, value=24, step=1)
    loan_age_years = st.number_input("Loan Age (years)", min_value=0, value=2, step=1)
    monthly_payment = st.number_input("Monthly Payment", min_value=0, value=1000, step=50)
    total_payment = st.number_input("Total Payment", min_value=0, value=24000, step=100)
    interest_amount = st.number_input("Interest Amount", min_value=0, value=5000, step=100)
    cur_principal = st.number_input("Current Principal", min_value=0, value=100000, step=1000)

# Additional required fields
postal_code = st.text_input("Postal Code", value="missing")
occupancy = st.selectbox("Occupancy (Owner, Tenant)", ["Owner", "Tenant", "missing"])
maturity_date = st.date_input("Maturity Date", value=pd.to_datetime("2024-09-19"))
first_payment_date = st.date_input("First Payment Date", value=pd.to_datetime("2024-09-19"))
emi = st.number_input("EMI", min_value=0.0, value=0.0, step=100.0)
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000, step=100)
prepayment = st.number_input("Prepayment", min_value=0, value=0, step=100)
mip = st.number_input("MIP", min_value=0.0, value=0.0, step=0.01)
ocltv = st.number_input("OCLTV", min_value=0.0, value=0.0, step=0.01)

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
            'postal_code': postal_code,
            'occupancy': occupancy,
            'maturity_date': maturity_date,
            'first_payment_date': first_payment_date,
            'emi': emi,
            'monthly_income': monthly_income,
            'prepayment': prepayment,
            'MIP': mip,
            'OCLTV': ocltv,
            'FirstTimeHomebuyer': 0,  # Placeholder
            'Channel': 'missing',  # Placeholder
            'ProductType': 'missing',  # Placeholder
            'PropertyState': 'missing',  # Placeholder
            'PropertyType': 'missing',  # Placeholder
            'LoanPurpose': 'missing',  # Placeholder
            'ServicerName': 'missing',  # Placeholder
            'SellerName_FT': 0,  # Placeholder
            'SellerName_AC': 0,  # Placeholder
            'SellerName_CO': 0,  # Placeholder
            'SellerName_NO': 0,  # Placeholder
            'SellerName_BA': 0,  # Placeholder
            'SellerName_FL': 0,  # Placeholder
            'SellerName_BI': 0,  # Placeholder
            'SellerName_PR': 0,  # Placeholder
            'SellerName_OL': 0,  # Placeholder
            'SellerName_FI': 0,  # Placeholder
            'SellerName_CR': 0,  # Placeholder
            'SellerName_PN': 0,  # Placeholder
            'SellerName_GM': 0,  # Placeholder
            'SellerName_ST': 0,  # Placeholder
            'SellerName_WA': 0,  # Placeholder
            'SellerName_Unknown': 0,  # Placeholder
            'SellerName_Ot': 0,  # Placeholder
            'SellerName_RE': 0,  # Placeholder
            'MSA': 0,  # Placeholder
            'NumBorrowers': 1,  # Placeholder
            'Units': 1,  # Placeholder
            'OrigLoanTerm': 30  # Placeholder
    }])

# Display the DataFrame to the user
st.write("### User Input DataFrame:")
st.write(user_input_df)

# Predict and display results
if st.button('Predict Classification and Regression'):
    try:
        # Get classification prediction (Ever Delinquent)
        y_class_pred = logistic_model.predict(user_input_df)
        st.write(f"Classification Prediction (Ever Delinquent): {y_class_pred[0]}")

        # Get regression prediction (Prepayment)
        y_reg_pred = linear_model.predict(user_input_df)
        st.write(f"Regression Prediction (Prepayment): {y_reg_pred[0]}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
