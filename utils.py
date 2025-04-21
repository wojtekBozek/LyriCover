import json
import logging
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from text_utils import compute_cosine_similarity, generate_lyrics
from hpcp_utils import extract_tonal_features
from config import whisper_size, training_pairs_number

# Parameters
MAX_PAIRS = training_pairs_number

def read_metadata(json_path):
    logging.info(f"Reading metadata from {json_path}...")
    try:
        data = json.load(open(json_path))
        logging.info(f"Successfully loaded metadata. Total records: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Failed to read metadata: {e}")
        raise

def load_whisper_model():
    import whisper  # Local import to keep main.py clean
    logging.info("Loading Whisper model...")
    return whisper.load_model(whisper_size, device="cuda")

def generate_pairs(metadata, max_pairs=MAX_PAIRS):
    logging.info("Generating random pairs of songs for cover classification...")
    
    # Group songs by clique_id
    clique_songs = defaultdict(list)
    for item in metadata:
        clique_id = item['utt'].split('_')[0]  # Extract clique ID from item
        clique_songs[clique_id].append(item)

    # Initialize lists to store pairs
    cover_pairs = []
    not_cover_pairs = []

    # Generate random cover pairs from songs within the same clique
    for clique_id, songs in clique_songs.items():
        if len(cover_pairs) >= max_pairs / 2:
            break        
        if len(songs) >= 2:  # Only consider cliques with at least 2 songs
            random.shuffle(songs)  # Shuffle songs in the clique
            for i in range(len(songs)):
                for j in range(i + 1, len(songs)):
                    if len(cover_pairs) < max_pairs / 2:
                        cover_pairs.append((songs[i], songs[j], 1))  # Add cover pair
                    else:
                        break
                if len(cover_pairs) >= max_pairs / 2:
                    break

    # Generate random non-cover pairs from different cliques
    clique_list = list(clique_songs.values())
    while len(not_cover_pairs) < max_pairs / 2:
        # Select two different cliques at random
        clique_a, clique_b = random.sample(clique_list, 2)

        # Randomly pick a song from each clique
        song_a = random.choice(clique_a)
        song_b = random.choice(clique_b)
        
        not_cover_pairs.append((song_a, song_b, 0))  # Add non-cover pair

    # Shuffle cover and non-cover pairs to ensure randomness
    random.shuffle(cover_pairs)
    random.shuffle(not_cover_pairs)

    logging.info(f"Generated {len(cover_pairs)} cover pairs and {len(not_cover_pairs)} not_cover pairs.")
    
    # Return the generated cover and non-cover pairs
    return cover_pairs+not_cover_pairs



def extract_pair_features(pairs, model):
    logging.info("Extracting features for pairs...")
    features, labels = [], []
    for pair in pairs:
        song_a, song_b, label = pair
        audio_a, audio_b = song_a['wav'], song_b['wav']
        lyrics_a, is_instrumental_a = generate_lyrics(audio_a, model)
        lyrics_b, is_instrumental_b = generate_lyrics(audio_b, model)
        tonal_features_a = extract_tonal_features(audio_a)
        tonal_features_b = extract_tonal_features(audio_b)
        tonal_similarity = np.dot(tonal_features_a, tonal_features_b) / (
            np.linalg.norm(tonal_features_a) * np.linalg.norm(tonal_features_b)
        )
        lyrics_similarity = compute_cosine_similarity(lyrics_a, lyrics_b) if not (
            is_instrumental_a or is_instrumental_b) else 0.1
        features.append(np.array([tonal_similarity, lyrics_similarity]))
        labels.append(label)
    logging.info("Feature extraction for pairs completed.")
    return np.array(features), np.array(labels)

def split_data(features, labels, test_percentage = 0.2):
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    logging.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

def cover_stats(X_train, y_train, X_test, y_test):
    train_cover_count, train_not_cover_count = np.sum(y_train == 1), np.sum(y_train == 0)
    test_cover_count, test_not_cover_count = np.sum(y_test == 1), np.sum(y_test == 0)
    logging.info(f"Train set - Cover: {train_cover_count}, Not-cover: {train_not_cover_count}")
    logging.info(f"Test set - Cover: {test_cover_count}, Not-cover: {test_not_cover_count}")
