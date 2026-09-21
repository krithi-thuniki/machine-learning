import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("height_weight.csv")

X = df[["Height"]]
y = df["Weight"]

model = LinearRegression()
model.fit(X, y)

st.title("Height → Weight Predictor")

height = st.number_input(
    "Enter your height (cm)",
    min_value=100.0,
    max_value=220.0,
    value=170.0
)

if st.button("Predict Weight"):
    prediction = model.predict([[height]])

    st.success(f"Predicted weight: {prediction[0]:.2f} kg")