import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Classification App",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS (Modern Aesthetic with Shadows and Card Layouts) ---
st.markdown("""
    <style>
    /* Global background and typography */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
    }
    .header-card h1 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .header-card p {
        color: #e0e6ed;
        font-size: 1.1rem;
    }

    /* Input Form Container Card */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
    }

    /* Card styling for elements */
    .css-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    /* Success / Prediction Output Box */
    .result-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border-left: 6px solid #2a5298;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin-top: 1.5rem;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }

    /* Custom Submit Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(42, 82, 152, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 15px rgba(42, 82, 152, 0.5);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Trained Model ---
@st.cache_resource
def load_model():
    with open("random.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `random.pkl`: {e}")
    st.stop()

# --- Header Section ---
st.markdown("""
    <div class="header-card">
        <h1>Predictive Intelligence Portal</h1>
        <p>Enter customer demographics to classify target customer status</p>
    </div>
""", unsafe_allow_html=True)

# --- Input Form ---
with st.form(key="prediction_form"):
    st.subheader("Customer Profile Inputs")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=0
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            options=["Single", "Married", "Divorced"],
            index=0
        )
        
        occupation = st.selectbox(
            "Occupation",
            options=["Student", "Employee", "Self Employed", "House wife"],
            index=1
        )

    with col2:
        monthly_income = st.selectbox(
            "Monthly Income",
            options=["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"],
            index=2
        )
        
        educational_qualifications = st.selectbox(
            "Educational Qualifications",
            options=["Uneducated", "School", "Graduate", "Post Graduate", "Ph.D"],
            index=2
        )
        
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3, step=1)
        
        customer_type = st.selectbox(
            "Customer Type",
            options=["Regular", "Occasional", "New"],
            index=0
        )

    submit_button = st.form_submit_button(label="Generate Prediction")

# --- Model Inference ---
if submit_button:
    # Construct DataFrame with explicit Category dtypes matching the dataset features
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Marital Status": marital_status,
        "Occupation": occupation,
        "Monthly Income": monthly_income,
        "Educational Qualifications": educational_qualifications,
        "Family size": family_size,
        "Customer Type": customer_type
    }])

    # Convert object columns to 'category' type as requested
    categorical_cols = [
        "Gender", "Marital Status", "Occupation", 
        "Monthly Income", "Educational Qualifications", "Customer Type"
    ]
    
    for col in categorical_cols:
        input_data[col] = input_data[col].astype("category")

    try:
        prediction = model.predict(input_data)[0]
        
        # Display Result with Shadow Effect Card
        st.markdown(f"""
            <div class="result-card">
                <h3>Prediction Result</h3>
                <div class="result-text">{prediction}</div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as err:
        st.error(f"Inference Error: {err}")
