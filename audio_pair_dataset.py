import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import numpy as np
import tempfile
import librosa

from hpcp_utils import extract_tonal_features
from text_utils import compute_cosine_similarity, generate_lyrics, load_lyrics, is_empty_or_stop_words, save_lyrics


class AudioPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, lyrics_model, instrumental_threshold, augmentation_fn=None, lyrics_dir="lyrics", load_lyrics=True, save_lyrics=True):
        self.pairs = pairs
        self.lyrics_model = lyrics_model
        self.instrumental_threshold = instrumental_threshold
        self.augmentation_fn = augmentation_fn
        self.lyrics_dir = lyrics_dir
        self.load_lyrics = load_lyrics
        self.save_lyrics = save_lyrics

    def __len__(self):
        return len(self.pairs)

    def gen_lyrics(self, audio_path, filepath, save=True):
        lyrics, is_instrumental = generate_lyrics(
            audio_path, instrumental_threshold=self.instrumental_threshold, model=self.lyrics_model
        )
        # Check if lyrics are meaningful
        if not lyrics or is_empty_or_stop_words(lyrics):
            logging.warning(f"Lyrics for {audio_path} are empty or contain only stop words. Treating as instrumental.")
            lyrics, is_instrumental = None, True
        if save is True and lyrics is not None:
            save_lyrics(filepath, lyrics)
        if save is True and lyrics is None:
            lyrics = ""
            save_lyrics(filepath, lyrics)
        return lyrics, is_instrumental

    def extract_tonal_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            if self.augmentation_fn:
                y = self.augmentation_fn(y, sr)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            tonal_features = np.mean(chroma, axis=1)
            return tonal_features
        except FileNotFoundError:
            logging.error(f"File not found: {audio_path}")
            return np.zeros(12)
        except librosa.util.exceptions.ParameterError as e:
            logging.error(f"Parameter error when processing {audio_path}: {e}")
            return np.zeros(12)
        except ValueError as e:
            logging.error(f"Value error when processing {audio_path}: {e}")
            return np.zeros(12)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        song_a, song_b, label = pair
        audio_a, audio_b = song_a['wav'], song_b['wav']
        lyrics_path_a = os.path.join(self.lyrics_dir, f"{os.path.basename(audio_a)}.txt")
        lyrics_path_b = os.path.join(self.lyrics_dir, f"{os.path.basename(audio_b)}.txt")

        # Extract features
        if self.load_lyrics and os.path.exists(lyrics_path_a):
            #logging.info(f"Loading lyrics from {lyrics_path_a}")
            lyrics_a, is_inst_a = load_lyrics(lyrics_path_a), is_empty_or_stop_words(load_lyrics(lyrics_path_a))
            tonal_a = self.extract_tonal_features(audio_a)
        else:
            #logging.info(f"Can not load lyrics from {lyrics_path_a}, generating...")
            lyrics_a, is_inst_a = self.gen_lyrics(audio_a, lyrics_path_a, self.save_lyrics)
            tonal_a = self.extract_tonal_features(audio_a)
        if self.load_lyrics and os.path.exists(lyrics_path_b):
            #logging.info(f"Loading lyrics from {lyrics_path_b}")
            lyrics_b, is_inst_b = load_lyrics(lyrics_path_b), is_empty_or_stop_words(load_lyrics(lyrics_path_b))
            tonal_b = self.extract_tonal_features(audio_b)
        else:
            #logging.info(f"Can not load lyrics from {lyrics_path_b}, generating...")
            lyrics_b, is_inst_b= self.gen_lyrics(audio_b, lyrics_path_b, self.save_lyrics)
            tonal_b = self.extract_tonal_features(audio_b)

        # Compute similarity
        tonal_similarity = np.dot(tonal_a, tonal_b) / (np.linalg.norm(tonal_a) * np.linalg.norm(tonal_b))
        lyrics_similarity = compute_cosine_similarity(lyrics_a, lyrics_b) if not (is_inst_a or is_inst_b) else 0.1

        feature_vector = torch.tensor([tonal_similarity, lyrics_similarity], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return feature_vector, label