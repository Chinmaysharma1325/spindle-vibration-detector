
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Spindle Vibration Fault Detector", layout="centered")

st.title("🛠️ Spindle Vibration Anomaly Detection")
st.markdown("Detect if the vibration signals are normal or faulty using an AI model.")

# Sidebar info
st.sidebar.title("📊 Input Vibration Data")
st.sidebar.write("Enter RMS vibration values in 3 axes:")

x = st.sidebar.slider("X-axis Vibration", -5.0, 10.0, 0.0)
y = st.sidebar.slider("Y-axis Vibration", -5.0, 10.0, 0.0)
z = st.sidebar.slider("Z-axis Vibration", -5.0, 10.0, 0.0)

input_data = np.array([[x, y, z]])

# Train a simple model (simulated)
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 3))
faulty_data = np.random.normal(loc=5, scale=1.5, size=(100, 3))
X = np.vstack((normal_data, faulty_data))

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

prediction = model.predict(input_data)[0]

if prediction == 1:
    st.success("✅ Status: NORMAL")
else:
    st.error("🚨 Status: FAULT DETECTED")

st.markdown("---")
st.caption("Project by: Chinmay Sharma | GTU AI for Manufacturing")
