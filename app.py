import pickle
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    """Load the classifier and the exact CountVectorizer it was trained with."""
    with open(BASE_DIR / 'spam.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(BASE_DIR / 'vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    return model, cv


st.title("SMS spam Detection Model")

try:
    model, cv = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing model file: {Path(e.filename).name}. "
        "Run 'Spam Detector.ipynb' from top to bottom to train the model and create it."
    )
    st.stop()

st.write("This is a Machine Learning application to detect SMS as spam or not spam.")
user_input = st.text_area("Enter an SMS to predict whether it's spam or not spam", height=150)

if user_input:
    st.caption(f"{len(user_input)} characters, {len(user_input.split())} words")

if st.button("Predict"):
    if user_input:
        vectorized_data = cv.transform([user_input])
        result = model.predict(vectorized_data)
        if result[0]==0:
            st.write("The SMS is not spam")
        else:
            st.write("The SMS is spam")
    else:
        st.write("Please type SMS to predict whether it's spam or not spam")
