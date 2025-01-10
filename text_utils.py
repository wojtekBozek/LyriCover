from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import logging

def preprocess_text(text):
    return text.lower()

def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(ngram_range=(3, 3))
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]


def generate_lyrics(file_path, model, instrumental_threshold):
    logging.info(f"Generating lyrics for: {file_path}")
    try:
        transcription = model.transcribe(file_path)['text'].strip().lower()
        is_instrumental = len(transcription.split()) < instrumental_threshold
        return transcription, is_instrumental
    except ValueError as e:
        logging.error(f"Value error when processing {file_path}: {e}")
        return np.zeros(12)
    except RuntimeError as e:
        logging.error(f"Failed to process file {file_path}: {e}")
    except NotImplementedError as e:
        logging.error(f"Using not implemented feature of Whisper: {e}")