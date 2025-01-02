from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    return text.lower()

def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(ngram_range=(3, 3))
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
