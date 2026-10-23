import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("Heart Disease Prediction System")
st.write(
    "Enter the four attributes below to generate "
    "a prediction from the trained machine-learning model."
)

age = st.number_input(
    "Age (years)", min_value=1, max_value=120, value=45
)

sex_label = st.selectbox(
    "Sex", ["Male", "Female"]
)
sex = 1 if sex_label == "Male" else 0

cp_label = st.selectbox(
    "Chest Pain Type",
    [
        "1 - Typical angina",
        "2 - Atypical angina",
        "3 - Non-anginal pain",
        "4 - Asymptomatic"
    ]
)
cp = int(cp_label[0])

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50, max_value=250, value=150
)

if st.button("Predict"):
    inputs = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    prediction = int(model.predict(inputs)[0])

    if prediction == 1:
        st.error("Prediction: Disease Present")
    else:
        st.success("Prediction: No Disease Detected")

    st.caption(
        "Educational demonstration only. This prediction "
        "is not a medical diagnosis or a substitute for "
        "professional medical advice."
    )
