fimport streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === NLTK Setup (only needed once) ===
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Load model and vectorizer ===
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Preprocessing Function ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# === Streamlit UI ===
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♀️")
st.title("🕵️‍♀️ Fake Review Detection System")
st.markdown("Detect whether a product review is genuine or fake using NLP and Machine Learning.")

# Input
user_input = st.text_area("✍️ Enter a product review:", height=150)

if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]

        if result == "OR":
            st.success("✅ **Genuine Review Detected**")
        else:
            st.error("❌ **Fake Review Detected**")
