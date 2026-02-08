import streamlit as st
st.write("App Started Successfully")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# PAGE TITLE
# ----------------------------
st.title("❤️ Love Compatibility Predictor")
st.write("AI Based Valentine Compatibility Prediction")

# ----------------------------
# CREATE SYNTHETIC DATASET
# ----------------------------
np.random.seed(42)

size = 400

df = pd.DataFrame({
    "interest_similarity": np.random.randint(40, 100, size),
    "communication": np.random.randint(40, 100, size),
    "trust": np.random.randint(40, 100, size),
    "time_spent": np.random.randint(30, 100, size),
    "love_language": np.random.randint(30, 100, size)
})

df["compatibility"] = (
    df["interest_similarity"] * 0.25 +
    df["communication"] * 0.20 +
    df["trust"] * 0.30 +
    df["time_spent"] * 0.15 +
    df["love_language"] * 0.10 +
    np.random.normal(0, 3, size)
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
X = df.drop("compatibility", axis=1)
y = df["compatibility"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# ----------------------------
# USER INPUT
# ----------------------------
st.header("Enter Relationship Details")

interest = st.slider("Interest Similarity", 0, 100, 70)
communication = st.slider("Communication Level", 0, 100, 70)
trust = st.slider("Trust Level", 0, 100, 70)
time_spent = st.slider("Time Spent Together", 0, 100, 70)
love_language = st.slider("Love Language Match", 0, 100, 70)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Compatibility ❤️"):

    user_data = np.array([[interest, communication, trust, time_spent, love_language]])
    prediction = model.predict(user_data)[0]

    st.success(f"💖 Compatibility Score: {round(prediction,2)}%")

    # Feature Importance Graph
    st.subheader("Feature Importance")
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(X.columns, importance)
    plt.xticks(rotation=30)
    st.pyplot(fig)

