import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="CKD Prediction", layout="wide")

# 🎨 Styling
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
h1 {
    color: #1f77b4;
}
.section {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# 🏥 HEADER
st.markdown("<h1 style='text-align:center;'>🩺 CKD Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>AI-based Kidney Disease Detection Platform</p>", unsafe_allow_html=True)

st.markdown("---")

# 📐 LAYOUT
left, center, right = st.columns([1,2,1])

# 🖼️ LEFT SIDE (Kidney + Medical Icons)
with left:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966487.png", width=150)
    st.caption("Kidney Health")

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=120)
    st.caption("Blood Pressure")

# 📋 CENTER FORM
with center:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🧾 Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100)
        bp = st.number_input("Blood Pressure")
        sg = st.number_input("Specific Gravity")
        al = st.number_input("Albumin")
        su = st.number_input("Sugar")

    with col2:
        hemo = st.number_input("Hemoglobin")
        pcv = st.number_input("Packed Cell Volume")
        wc = st.number_input("WBC Count")
        rc = st.number_input("RBC Count")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict", use_container_width=True):
        features = np.array([[age, bp, sg, al, su, hemo, pcv, wc, rc]])
        prediction = model.predict(features)

        st.markdown("---")

        if prediction[0] == 0:
            st.error("⚠️ Chronic Kidney Disease Detected")

            st.markdown("### 💊 Recommended Care")
            st.markdown("""
            <div style="background-color:#ffe6e6; padding:15px; border-radius:10px;">
            ✔ Reduce salt intake<br>
            ✔ Control blood pressure<br>
            ✔ Drink water as advised<br>
            ✔ Avoid processed foods<br>
            ✔ Regular doctor consultation<br>
            ✔ Monitor kidney function<br>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("✅ No Kidney Disease Detected")

            st.markdown("### 🌿 Healthy Lifestyle Tips")
            st.markdown("""
            <div style="background-color:#e6ffe6; padding:15px; border-radius:10px;">
            ✔ Stay hydrated<br>
            ✔ Balanced diet<br>
            ✔ Exercise regularly<br>
            ✔ Avoid excess salt<br>
            ✔ Routine health checkups<br>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 🏥 RIGHT SIDE (Hospital + Health Icons)
with right:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209265.png", width=150)
    st.caption("Hospital Care")

    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=120)
    st.caption("Healthcare Monitoring")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>🏥 CKD ML Project | Smart Healthcare System</p>", unsafe_allow_html=True)