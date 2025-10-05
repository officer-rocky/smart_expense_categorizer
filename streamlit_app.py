# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"

st.title("💰 Smart Expense Categorizer")
st.write("Enter a transaction text to predict its category:")

transaction_text = st.text_input("Transaction:", placeholder="e.g., Swiggy 250")

if st.button("Predict Category"):
    if transaction_text.strip():
        response = requests.post(API_URL, json={"text": transaction_text})
        if response.status_code == 200:
            data = response.json()
            predicted = data["preds"][0]
            st.success(f"Predicted Category: **{predicted}**")

            st.write("Probabilities:")
            probs = data["probs"][0]
            classes = data["classes"]
            for cls, prob in zip(classes, probs):
                st.write(f"{cls}: {prob:.2%}")
        else:
            st.error("Error: Could not get response from the API.")
    else:
        st.warning("Please enter a transaction.")
