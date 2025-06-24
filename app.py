import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Setup NLTK ===
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# === Load Model and Vectorizer ===
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Text Preprocessing Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# === Heuristic Rule Check ===
def is_suspect_review(text):
    repeated_words = len(re.findall(r'\b(\w+)\b(?=.*\b\1\b)', text.lower()))
    emojis = len(re.findall(r'[^\w\s,]', text))
    all_caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    if repeated_words > 2 or emojis > 3 or all_caps_words > 2:
        return True
    return False

# === Streamlit UI ===
st.set_page_config(page_title="🕵️ Fake Review Detector", layout="centered")
st.title("🕵️‍♀️ Fake Review Detection System")
st.caption("Using NLP + Machine Learning to identify fake product reviews in real-time.")

# === User Input ===
st.subheader("✍️ Enter a product review:")
user_input = st.text_area(" ", height=140, placeholder="E.g. This product changed my life...")

# === Analyze Single Review ===
if st.button("🔍 Analyze Review") and user_input.strip():
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    suspicious = is_suspect_review(user_input)
    label = "❌ Possibly Fake (Suspicious Pattern Detected)" if suspicious and prediction == "OR" else ("✅ Genuine Review" if prediction == "OR" else "❌ Fake Review")
    color = "orange" if suspicious and prediction == "OR" else ("green" if prediction == "OR" else "red")

    st.markdown(f"### 🎯 <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Score:** `{round(confidence * 100, 2)}%`")
    st.progress(int(confidence * 100))

    # Show Influencing Words
    st.subheader("🧠 Influential Keywords Found")
    matched = [word for word in cleaned.split() if word in vectorizer.get_feature_names_out()]
    if matched:
        st.success("The model found these impactful words: `" + "`, `".join(matched) + "`")
    else:
        st.warning("No important words matched with the training dataset vocabulary.")

    # Download CSV result
    df = pd.DataFrame({
        "Review": [user_input],
        "Prediction": [label],
        "Confidence (%)": [round(confidence * 100, 2)]
    })
    st.download_button("📥 Download Result as CSV", df.to_csv(index=False).encode('utf-8'),
                       file_name="review_result.csv", mime='text/csv')

# === Batch Mode ===
with st.expander("📦 Analyze Multiple Reviews (Batch Mode)"):
    batch_input = st.text_area("Paste multiple reviews (one per line):", height=200)
    if st.button("📊 Analyze All"):
        st.subheader("📋 Batch Review Results")
        reviews = [r.strip() for r in batch_input.split("\n") if r.strip()]
        results = []
        for review in reviews:
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            label = "✅ Genuine" if pred == "OR" else "❌ Fake"
            results.append((review, label))

        for i, (rev, lbl) in enumerate(results):
            st.markdown(f"**{i+1}.** {rev}")
            st.markdown(f"🧾 Result: {lbl}")
            st.markdown("---")

        batch_df = pd.DataFrame(results, columns=["Review", "Prediction"])
        batch_csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Batch Results", data=batch_csv, file_name="batch_results.csv", mime='text/csv')
