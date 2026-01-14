import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("🩺 Diabetes Prediction App")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")

    # Replace 0 with NA for selected columns
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols] = df[cols].replace(0, pd.NA)

    # Fill missing values with column mean
    df.fillna(df.mean(), inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df

df = load_data()

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, scaler, accuracy

model, scaler, accuracy = train_model(df)

st.write(f"### 🔍 Model Accuracy: **{accuracy:.2f}**")

st.subheader("Enter Patient Details for Prediction")

# User Inputs (same order as features)
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 1, 300, 120)
bp = st.number_input("Blood Pressure", 1, 200, 70)
skin = st.number_input("Skin Thickness", 1, 99, 20)
insulin = st.number_input("Insulin Level", 1, 900, 80)
bmi = st.number_input("BMI", 1.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prepare input for prediction
if st.button("Predict Diabetes"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Scale input
    scaled_input = scaler.transform(user_input)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.write(f"### Probability of Diabetes: **{probability:.2f}**")

    if prediction == 1:
        st.error("🔴 High Chance of Diabetes")
    else:
        st.success("🟢 Low Chance of Diabetes")