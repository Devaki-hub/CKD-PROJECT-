import streamlit as st
import numpy as np
import pickle
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Title
st.markdown("<h1 style='text-align: center; color: blue;'>CKD Prediction System</h1>", unsafe_allow_html=True)

st.write("### Enter Patient Details")

# Inputs
age = st.number_input("Age", 1, 100)
bp = st.number_input("Blood Pressure", 50, 200)
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.slider("Albumin", 0, 5)
su = st.slider("Sugar", 0, 5)
hemo = st.number_input("Hemoglobin", 0.0, 20.0)
pcv = st.number_input("Packed Cell Volume", 0, 60)
wc = st.number_input("White Blood Cell Count", 0, 20000)
rc = st.number_input("Red Blood Cell Count", 0.0, 10.0)

# Prediction
if st.button("Predict"):

    data = np.array([[age, bp, sg, al, su, hemo, pcv, wc, rc]])

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("⚠️ CKD Detected")
    else:
        st.success("✅ No CKD")