import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('french'))
stemmer = SnowballStemmer('french')
nlp = spacy.load('fr_core_news_md')
def process_text(text):
    if text is None or pd.isna(text):
        return ''
    if not isinstance(text, str):
        text = str(text)
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    doc = nlp(" ".join(stemmed_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)




def clean_text(text):
    if text is None or pd.isna(text):
        return ''
    cleaned_text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)  # Keeps only letters and spaces
    return cleaned_text
