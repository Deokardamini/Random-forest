from pathlib import Path
import pickle
import streamlit as st

# Find the absolute path to the directory where app.py lives
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "RandomForest.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()
