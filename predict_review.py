import pandas as pd
from model_utils import load_model
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download only once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load model and vectorizer
model, tfidf = load_model()

while True:
    user_input = input("📝 Enter a review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 'CG':
        print("🔍 Review is: ✅ Genuine")
    else:
        print("🔍 Review is: ❌ Fake")
