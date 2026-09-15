import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Response Predictor",
    page_icon="🎯",
    layout="wide"
)

# --- CUSTOM CSS FOR ATTRACTIVE STYLING ---
st.markdown("""
    <style>
    /* Main Background Accent */
    .main {
        background-color: #f8f9fa;
    }
    /* Title Card Styling */
    .title-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .title-card h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .title-card p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    /* Prediction Box Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('random.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model 'random.pkl': {e}")
    st.stop()

# --- HEADER SECTION ---
st.markdown("""
    <div class="title-card">
        <h1>🎯 Customer Analytics Predictor</h1>
        <p>Enter the demographic details below to predict customer response.</p>
    </div>
""", unsafe_allow_html=True)

# --- FORM INPUTS SECTION ---
st.subheader("📋 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    occupation = st.selectbox("Occupation", ["Student", "Employed", "Self-Employed", "Unemployed"])

with col2:
    monthly_income = st.selectbox("Monthly Income", ["No Income", "Below 10,000", "10,001 - 25,000", "25,001 - 50,000", "More than 50,000"])
    education = st.selectbox("Educational Qualifications", ["Under Graduate", "Post Graduate", "School", "Uneducated"])
    family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3, step=1)
    customer_type = st.selectbox("Customer Type", ["New", "Existing", "Frequent", "Occasional"])

st.markdown("---")

# --- PREDICTION TRIGGER ---
if st.button("🔮 Predict Response"):
    
    # 1. Map Categorical inputs to numerical features if required by your pipeline,
    # or build the dataframe matching feature order: 
    # ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'Customer Type']
    
    input_data = pd.DataFrame([[
        age, gender, marital_status, occupation, 
        monthly_income, education, family_size, customer_type
    ]], columns=[
        'Age', 'Gender', 'Marital Status', 'Occupation', 
        'Monthly Income', 'Educational Qualifications', 'Family size', 'Customer Type'
    ])

    # Show a spinner effect during processing
    with st.spinner("Analyzing input parameters..."):
        time.sleep(0.8)  # Subtle pause for dynamic effect
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data).max() * 100 if hasattr(model, "predict_proba") else None
        except Exception as err:
            # Fallback if preprocessing is required prior to prediction
            st.error("Prediction failed. Make sure categorical inputs match the model's target encoder or preprocessor format.")
            st.exception(err)
            st.stop()

    # Visual Celebration Effect
    st.balloons()
    st.snow()

    # --- RESULT DISPLAY ---
    st.markdown("### 📊 Prediction Result")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if str(prediction).lower() in ["yes", "1", "true"]:
            st.success(f"### 🎉 Prediction: {prediction}")
            st.write("The customer is **highly likely** to convert or accept the offer.")
        else:
            st.warning(f"### ⚠️ Prediction: {prediction}")
            st.write("The customer is **unlikely** to convert or accept the offer.")

    with res_col2:
        if probability is not None:
            st.metric(label="Model Confidence", value=f"{probability:.2f}%")
