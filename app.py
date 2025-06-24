import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Download resources (only once) ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load model and vectorizer ---
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Init NLP tools ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# --- Heuristic checker for suspicious patterns ---
def is_suspect_review(text):
    repeated_words = len(re.findall(r'\b(\w+)\b(?=.*\b\1\b)', text.lower()))
    emojis = len(re.findall(r'[^\w\s,]', text))
    all_caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    if repeated_words > 2 or emojis > 3 or all_caps_words > 2:
        return True
    return False

# --- Streamlit App UI ---
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🕵️‍♀️ Fake Review Detection System")
st.markdown("Using NLP + Machine Learning to identify fake product reviews in real-time.")

# --- Input Section ---
st.subheader("📝 Enter a product review:")
user_input = st.text_area(" ", height=150)

if st.button("🔍 Analyze Review") and user_input.strip():
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    # Check with heuristic rules
    suspicious = is_suspect_review(user_input)

    # Labeling logic
    if suspicious and prediction == "OR":
        label = "❌ Possibly Fake (Suspicious Pattern Detected)"
        color = "orange"
    else:
        label = "Genuine ✅" if prediction == "OR" else "Fake ❌"
        color = "green" if prediction == "OR" else "red"

    # Display prediction
    st.markdown(f"### 🎯 Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Score:** {round(confidence * 100, 2)}%")
    st.progress(int(confidence * 100))

    # Highlight keywords
    st.subheader("🔍 Influential Keywords Found")
    matched_words = [w for w in cleaned.split() if w in vectorizer.get_feature_names_out()]
    if matched_words:
        st.success("The model found these impactful words: " + ", ".join(matched_words))
    else:
        st.warning("No important words found from trained data.")

    # CSV download
    result_df = pd.DataFrame({
        "Input Review": [user_input],
        "Prediction": [label],
        "Confidence (%)": [round(confidence * 100, 2)]
    })
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📅 Download Result as CSV", data=csv, file_name="review_result.csv", mime='text/csv')

# --- Optional Batch Review Checker ---
with st.expander("📦 Try Batch Mode (Multiple Reviews)"):
    batch_input = st.text_area("Paste multiple reviews here (one per line):", height=200)
    if st.button("📊 Analyze All"):
        st.subheader("📋 Batch Review Results:")
        batch_reviews = batch_input.strip().split("\n")
        results = []
        for review in batch_reviews:
            if review.strip():
                cleaned = clean_text(review)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                label = "Genuine ✅" if pred == "OR" else "Fake ❌"
                results.append((review, label))
                st.markdown(f"**Review:** {review}")
                st.markdown(f"🔎 Result: **{label}**")
                st.markdown("---")

        batch_df = pd.DataFrame(results, columns=["Review", "Prediction"])
        batch_csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📅 Download Batch Results as CSV", data=batch_csv, file_name="batch_reviews.csv", mime='text/csv')
