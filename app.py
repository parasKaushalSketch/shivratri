import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Configuration --- #
MODEL_PATH = 'ann_credit_score_model.keras'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'processed_credit_card_data.csv' # Used to get column names and default values

# Define the lists of numerical and categorical columns as used during training
NUMERICAL_COLS_FOR_SCALING = [
    'Age',
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance',
    'Credit_History_Age_Months'
]

CATEGORICAL_COLS_FOR_ENCODING = [
    'Month', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age', 
    'Payment_of_Min_Amount', 'Payment_Behaviour'
]

CREDIT_SCORE_MAPPING_REVERSE = {0: 'Poor', 1: 'Standard', 2: 'Good'}

# --- Load Artifacts --- #
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_ref = pd.read_csv(DATA_PATH) # Load for column names and default values
    return model, scaler, df_ref

model, scaler, df_ref = load_artifacts()

# --- Preprocessing Function (to mimic training data structure) --- #
def preprocess_input(user_data_dict, df_reference, scaler_obj):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_data_dict])

    # Ensure all original numerical and categorical columns are present
    # Initialize with median/mode values from df_reference
    for col in NUMERICAL_COLS_FOR_SCALING:
        if col not in input_df.columns:
            input_df[col] = df_reference[col].median()
    for col in CATEGORICAL_COLS_FOR_ENCODING:
        if col not in input_df.columns:
            input_df[col] = df_reference[col].mode()[0]

    # Order columns to match df_reference before one-hot encoding
    # Drop ID, Customer_ID, SSN explicitly if they exist in df_reference during setup
    # (assuming they were already dropped during preprocessing, but ensuring here)
    ref_cols_for_matching = df_reference.drop(columns=['ID', 'Customer_ID', 'SSN'], errors='ignore')
    ref_cols_for_matching = ref_cols_for_matching.drop(columns=['Credit_Score'], errors='ignore').columns
    
    input_df = input_df[ref_cols_for_matching]

    # Apply one-hot encoding for categorical features
    # Create a dummy row from df_reference to get all possible one-hot encoded columns
    temp_df_for_dummies = df_reference.drop(columns=['ID', 'Customer_ID', 'SSN', 'Credit_Score'], errors='ignore')
    temp_df_for_dummies = pd.get_dummies(temp_df_for_dummies, columns=CATEGORICAL_COLS_FOR_ENCODING, drop_first=True)
    all_model_columns = temp_df_for_dummies.columns

    # Apply get_dummies to the user input and reindex to match all_model_columns
    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS_FOR_ENCODING, drop_first=True)
    input_processed = input_encoded.reindex(columns=all_model_columns, fill_value=0)

    # Scale numerical features
    input_processed[NUMERICAL_COLS_FOR_SCALING] = scaler_obj.transform(input_processed[NUMERICAL_COLS_FOR_SCALING])

    return input_processed

# --- Streamlit UI --- #
st.set_page_config(page_title="Credit Score Predictor", layout="wide")
st.title("Credit Score Prediction for Credit Card Customers")
st.markdown("Enter customer details to predict their credit score.")

with st.sidebar:
    st.header("Input Features")
    
    # Numerical Inputs
    age = st.slider("Age", min_value=18, max_value=80, value=df_ref['Age'].median().astype(int))
    annual_income = st.number_input("Annual Income ($")", min_value=0.0, value=df_ref['Annual_Income'].median())
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary ($")", min_value=0.0, value=df_ref['Monthly_Inhand_Salary'].median())
    num_bank_accounts = st.slider("Number of Bank Accounts", min_value=0, max_value=10, value=df_ref['Num_Bank_Accounts'].median().astype(int))
    num_credit_card = st.slider("Number of Credit Cards", min_value=0, max_value=10, value=df_ref['Num_Credit_Card'].median().astype(int))

    # Categorical Inputs (using unique values from reference data)
    occupation = st.selectbox("Occupation", options=df_ref['Occupation'].unique(), index=list(df_ref['Occupation'].unique()).index(df_ref['Occupation'].mode()[0]))
    credit_mix = st.selectbox("Credit Mix", options=df_ref['Credit_Mix'].unique(), index=list(df_ref['Credit_Mix'].unique()).index(df_ref['Credit_Mix'].mode()[0]))
    payment_of_min_amount = st.selectbox("Payment of Minimum Amount", options=df_ref['Payment_of_Min_Amount'].unique(), index=list(df_ref['Payment_of_Min_Amount'].unique()).index(df_ref['Payment_of_Min_Amount'].mode()[0]))

# --- Prediction Button --- #
if st.button("Predict Credit Score"):
    user_input = {
        'Age': age,
        'Annual_Income': annual_income,
        'Monthly_Inhand_Salary': monthly_inhand_salary,
        'Num_Bank_Accounts': num_bank_accounts,
        'Num_Credit_Card': num_credit_card,
        'Occupation': occupation,
        'Credit_Mix': credit_mix,
        'Payment_of_Min_Amount': payment_of_min_amount,
    }

    # Fill in default values for other numerical columns not exposed in UI for simplicity
    for col in NUMERICAL_COLS_FOR_SCALING:
        if col not in user_input:
            user_input[col] = df_ref[col].median()
            
    # Fill in default values for other categorical columns not exposed in UI
    for col in CATEGORICAL_COLS_FOR_ENCODING:
        if col not in user_input:
            user_input[col] = df_ref[col].mode()[0]
            
    # Ensure 'Month' is handled, as it's often the first categorical feature
    if 'Month' not in user_input:
        user_input['Month'] = df_ref['Month'].mode()[0] # Example: January
    
    # Create a DataFrame for a single prediction
    # The preprocess_input function will ensure the structure is correct
    processed_user_input = preprocess_input(user_input, df_ref, scaler)
    
    # Make prediction
    prediction_proba = model.predict(processed_user_input)
    predicted_class = np.argmax(prediction_proba, axis=1)[0]
    predicted_credit_score = CREDIT_SCORE_MAPPING_REVERSE[predicted_class]

    st.success(f"The Predicted Credit Score is: **{predicted_credit_score}**")
    st.write(f"Prediction Probability: {prediction_proba[0][predicted_class]:.2f}")
