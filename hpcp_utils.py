import logging
import numpy as np
import librosa

def extract_tonal_features(file_path):
    try:
    
        y, sr = librosa.load(file_path, sr=None)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        tonal_features = np.mean(chroma, axis=1)
        
        return tonal_features
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return np.zeros(12)
    except librosa.util.exceptions.ParameterError as e:
        logging.error(f"Parameter error when processing {file_path}: {e}")
        return np.zeros(12)
    except ValueError as e:
        logging.error(f"Value error when processing {file_path}: {e}")
        return np.zeros(12)


