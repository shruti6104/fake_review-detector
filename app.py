# ✅ Final Improved `app.py` with all suggested features

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from fpdf import FPDF

# === Load Model & Vectorizer ===
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Initialize Preprocessing Tools ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Cleaning Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# === Generate PDF ===
def generate_pdf(review, prediction, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fake Review Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(200, 10, txt=f"Review: {review}")
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
    pdf.output("review_report.pdf")

# === Streamlit UI ===
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🕵️ Fake Review Detection System")
st.markdown("Detect whether a product review is Genuine or Fake using NLP + ML")

user_input = st.text_area("📝 Enter a product review:")

if st.button("🔍 Analyze Review") and user_input:
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence = max(probability)

    if result == 'OR':
        st.error("❌ Fake Review Detected")
        st.markdown("### 💡 Suggestions to Write a Genuine Review:")
        st.markdown("""
        - Mention specific product features  
        - Share actual usage experience  
        - Avoid too many emotional/exaggerated words
        """)
    else:
        st.success("✅ Genuine Review Detected")

    st.markdown(f"### 🔢 Confidence: **{confidence * 100:.1f}%**")

    # Pie Chart
    fig, ax = plt.subplots()
    labels = ['Genuine', 'Fake']
    colors = ['green', 'red']
    ax.pie(probability, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Download Button
    generate_pdf(user_input, 'Genuine' if result != 'OR' else 'Fake', confidence)
    with open("review_report.pdf", "rb") as f:
        st.download_button("📥 Download Result Report", f, "review_report.pdf")
