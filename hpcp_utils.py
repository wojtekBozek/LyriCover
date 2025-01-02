import logging
import numpy as np
import librosa

def extract_tonal_features(file_path):
    logging.info(f"Extracting tonal features for: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        tonal_features = np.mean(chroma, axis=1)
        return tonal_features
    except Exception as e:
        logging.error(f"Error extracting tonal features for {file_path}: {e}")
        return np.zeros(12)

def generate_lyrics(file_path, model, instrumental_threshold):
    logging.info(f"Generating lyrics for: {file_path}")
    try:
        transcription = model.transcribe(file_path)['text'].strip().lower()
        is_instrumental = len(transcription.split()) < instrumental_threshold
        return transcription, is_instrumental
    except Exception as e:
        logging.error(f"Error generating lyrics for {file_path}: {e}")
        return "", False
