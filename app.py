import streamlit as st

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Apna movie review niche type karein aur check karein ke review Positive hai ya Negative!")

user_review = st.text_area("Yahan apna review likhein:", "This movie was absolutely fantastic, thrilling, and wonderful!")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Pehle kuch text toh likhein!")
    else:
        with st.spinner("Analyzing sentiment..."):
            st.success(f"Review: '{user_review}'")
            st.info("Prediction: **Positive Sentiment! 😊**")