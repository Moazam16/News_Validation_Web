import streamlit as st
from vectorizer import transform_user_input
import joblib

model = joblib.load("model.pkl")
st.title("News Detection System")

user_text = st.text_area("Enter News")
if st.button("Predict"):
    final_input = transform_user_input(user_text)
    result = model.predict(final_input)[0]
    st.success("Real News" if result == 1 else " Fake News")
