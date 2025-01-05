import logging
import numpy as np
import librosa

import librosa
import numpy as np
import logging

def extract_tonal_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Compute mean chroma features across time
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