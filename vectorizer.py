import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
import numpy as np
from scipy.sparse import hstack
import nltk

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Function to map POS tags
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to clean, tokenize, remove stopwords, lemmatize
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized)

# Sentiment score
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Final vectorization function (to use in Streamlit app)
def transform_user_input(user_text):
    cleaned_text = preprocess_text(user_text)
    tfidf_features = vectorizer.transform([cleaned_text])  # shape (1, N)
    
    # Extra features
    sentiment = get_sentiment_score(cleaned_text)
    length = len(cleaned_text.split())
    extra_features = np.array([[sentiment, length]])  # shape (1, 2)

    # Combine features
    final_input = hstack([tfidf_features, extra_features])
    return final_input
