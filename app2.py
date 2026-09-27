import streamlit as st
import numpy as np
import pickle
import os

# load model
model = pickle.load(open("model.pkl", "rb"))

st.title("CKD Prediction System")

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

age = st.number_input("Age")
bp = st.number_input("Blood Pressure")
hemo = st.number_input("Hemoglobin")

if st.button("Predict"):
    data = np.array([[age, bp, hemo]])
    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("CKD Detected")
    else:
        st.success("No CKD")