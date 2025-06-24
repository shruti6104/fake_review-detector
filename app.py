import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Init NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# UI Layout
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🕵️‍♀️ Fake Review Detection System")
st.markdown("Detect whether a product review is **Genuine** or **Fake** using NLP + Machine Learning.")

# Input Section
st.subheader("📝 Enter a product review:")
user_input = st.text_area(" ", height=150)

if st.button("🔍 Analyze Review") and user_input.strip():
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    # Labeling
    label = "Genuine ✅" if prediction == "OR" else "Fake ❌"
    color = "green" if prediction == "OR" else "red"

    # Output
    st.markdown(f"### 🎯 Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** {round(confidence * 100, 2)}%")
    st.progress(int(confidence * 100))

    # Word Highlights
    st.subheader("🔍 Key Words in Your Review")
    matched_words = [w for w in cleaned.split() if w in vectorizer.get_feature_names_out()]
    if matched_words:
        st.success("These keywords influenced the model: " + ", ".join(matched_words))
    else:
        st.warning("No important words found from trained data.")

    # CSV Download
    result_df = pd.DataFrame({
        "Input Review": [user_input],
        "Prediction": [label],
        "Confidence (%)": [round(confidence * 100, 2)]
    })
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Result as CSV", data=csv, file_name="review_result.csv", mime='text/csv')

# Optional: Batch Review Checker
with st.expander("📦 Try Batch Mode (Multiple Reviews)"):
    batch_input = st.text_area("Paste multiple reviews here (one per line):", height=200)
    if st.button("📊 Analyze All"):
        st.subheader("📋 Batch Review Results:")
        batch_reviews = batch_input.strip().split("\n")
        for review in batch_reviews:
            if review.strip():
                cleaned = clean_text(review)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                label = "Genuine ✅" if pred == "OR" else "Fake ❌"
                st.markdown(f"**Review:** {review}")
                st.markdown(f"🔎 Result: **{label}**")
                st.markdown("---")
