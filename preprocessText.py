from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import spacy
import re
from rapidfuzz import process, fuzz

nltk.download('stopwords')
stop_words = list(stopwords.words('french'))
stop_words+=list(stopwords.words('english'))
stop_words=set(stop_words)
stemmer = SnowballStemmer('french')
nlp = spacy.load('fr_core_news_md')

# print(stop_words)
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
    cleaned_text = re.sub(r'[^a-zA-ZÀ-ÿçîïôàûùœêëèé\s]', ' ', text)  # Keeps all French letters and spaces
    return cleaned_text


def remove_adjectives(text):
    if text is None or pd.isna(text):
        return ''
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.pos_ != 'ADJ']
    return " ".join(filtered_tokens)


def get_most_frequent_words_by_cluster(df, cluster_col, text_col, top_n=30):
    if cluster_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Columns '{cluster_col}' or '{text_col}' not found in DataFrame.")

    cluster_groups = df.groupby(cluster_col)[text_col].apply(lambda x: ' '.join(x.dropna().astype(str)))
    frequent_words_by_cluster = {}

    for cluster, text in cluster_groups.items():
        vectorizer = CountVectorizer(stop_words=list(stop_words))  # Add 'english' if the text is in English
        X = vectorizer.fit_transform([text])
        word_counts = X.toarray().sum(axis=0)
        word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))
        most_common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        frequent_words_by_cluster[cluster] = most_common_words

    return frequent_words_by_cluster



def remove_non_significant_words(text):
    non_significant_words = {'france', 'lyon', 'img', 'jpg', 'jpeg','dsc','ddc'}
    if text is None or pd.isna(text):
        return ''
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in non_significant_words]
    return " ".join(filtered_tokens)



def compute_tfidf_by_cluster(df, cluster_col, text_col, top_n=10):
    if cluster_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Columns '{cluster_col}' or '{text_col}' not found in DataFrame.")
    
    # Grouping the text data by cluster
    cluster_groups = df.groupby(cluster_col)[text_col].apply(lambda x: ' '.join(x.dropna().astype(str)))

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=list(stop_words))  # Add French stopwords if needed
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_groups)

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Extract top words per cluster
    tfidf_scores_by_cluster = {}
    for idx, cluster in enumerate(cluster_groups.index):
        scores = tfidf_matrix[idx].toarray().flatten()
        word_score_pairs = list(zip(feature_names, scores))
        sorted_words = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)[:top_n]
        tfidf_scores_by_cluster[cluster] = sorted_words

    return tfidf_scores_by_cluster


def compute_tfidf_by_cluster(df, cluster_col,col1,col2, top_n=10):
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in DataFrame.")
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Columns col1 or col2 not found in DataFrame.")
    
    df['combined_text'] = df[[col1, col2]].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    cluster_groups = df.groupby(cluster_col)['combined_text'].apply(lambda x: ' '.join(x))
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_groups)
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores_by_cluster = {}
    for idx, cluster in enumerate(cluster_groups.index):
        scores = tfidf_matrix[idx].toarray().flatten()
        word_score_pairs = list(zip(feature_names, scores))
        sorted_words = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)[:top_n]
        tfidf_scores_by_cluster[cluster] = sorted_words
    
    return tfidf_scores_by_cluster


def reverse(df, cluster_col, col1, col2, tokenized_clusters, max_words=10, similarity_threshold=40):
    reversed_clusters = {}
    for cluster_id, tokens in tokenized_clusters.items():
        cluster_data = df[df[cluster_col] == cluster_id]
        original_text = ' '.join(cluster_data[col1].fillna('') + ' ' + cluster_data[col2].fillna(''))
        original_words = set(original_text.split())
        corrected_tokens = set()  
        for token in tokens:
            # Find the closest match using Levenshtein distance with a high threshold
            match_info = process.extractOne(token, original_words, scorer=fuzz.token_set_ratio	)
            if match_info:
                match, score, _ = match_info  # Extract the closest substring
                if score > similarity_threshold:
                    corrected_tokens.add(match)
                else:
                    corrected_tokens.add(token)
        reversed_clusters[cluster_id] = list(corrected_tokens)[:max_words]
    return reversed_clusters
