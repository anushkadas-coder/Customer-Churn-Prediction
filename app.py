import streamlit as st
import pandas as pd
import pickle

# --- LOAD THE TRAINED MODEL ---
@st.cache_data
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- APP TITLE AND DESCRIPTION ---
st.title('Customer Churn Prediction App 🔮')
st.markdown("This app uses a machine learning model to predict whether a customer is likely to churn. Enter the customer's details in the sidebar to get a prediction.")

# --- SIDEBAR FOR USER INPUTS ---
st.sidebar.header('Customer Details')

def user_input_features():
    # Input fields for all the features the model was trained on
    tenure = st.sidebar.slider('Tenure (months)', 1, 72, 24)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 118.0, 70.0)
    total_charges = st.sidebar.slider('Total Charges ($)', 18.0, 8684.0, 1400.0)
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', (0, 1))
    
    # Create a dictionary of the inputs
    data = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- DISPLAY USER INPUTS ---
st.subheader('Customer Details Entered:')
st.write(input_df)

# --- PREDICTION LOGIC ---
if st.button('Predict Churn'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.error('This customer is at HIGH RISK of churning.')
        st.write(f"Confidence Score: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success('This customer is likely to STAY.')
        st.write(f"Confidence Score: {prediction_proba[0][0]*100:.2f}%")