import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import numpy as np
from hpcp_utils import extract_tonal_features
from text_utils import compute_cosine_similarity, generate_lyrics


class CoverClassifierNN(nn.Module):
    def __init__(self):
        super(CoverClassifierNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)


class CoverClassifier:
    def __init__(self, instrumental_threshold=8, lyrics_model=None):
        self.instrumental_threshold = instrumental_threshold
        self.lyrics_model = lyrics_model
        self.nn_model = CoverClassifierNN()
        self.is_model_loaded = False
    
    def load_model(self, model_path="model.pth"):
        """Load the model once for predictions."""
        logging.info("Loading model...")
        self.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.nn_model.eval()
        self.is_model_loaded = True
        logging.info("Model loaded successfully.")


    def load_lyrics(self, filepath):
        """Load lyrics from a file if it exists."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                return file.read()
        return None

    def save_lyrics(self, filepath, lyrics):
        """Save lyrics to a file."""
        with open(filepath, 'w') as file:
            file.write(lyrics)

    def get_lyrics(self, audio_path, filepath, load_save=None):
        """Load, generate, or save lyrics for a given audio file."""
        lyrics = self.load_lyrics(filepath) if load_save == "load" else None
        if lyrics is None:
            logging.info(f"Lyrics not found for {audio_path}, generating...")
            lyrics, is_instrumental = generate_lyrics(
                audio_path, instrumental_threshold=self.instrumental_threshold, model=self.lyrics_model
            )
            if load_save == "save" and lyrics is not None:
                self.save_lyrics(filepath, lyrics)
        else:
            is_instrumental = False
        return lyrics, is_instrumental

    def extract_pair_features(self, pairs, load_save=None, lyrics_dir="lyrics"):
        logging.info("Extracting features for pairs...")
        features, labels = [], []
        for pair in pairs:
            song_a, song_b, label = pair
            audio_a, audio_b = song_a['wav'], song_b['wav']
            lyrics_path_a = os.path.join(lyrics_dir, f"{os.path.basename(audio_a)}.txt")
            lyrics_path_b = os.path.join(lyrics_dir, f"{os.path.basename(audio_b)}.txt")

            # Get lyrics and instrumental status for each song
            lyrics_a, is_instrumental_a = self.get_lyrics(audio_a, lyrics_path_a, load_save)
            lyrics_b, is_instrumental_b = self.get_lyrics(audio_b, lyrics_path_b, load_save)

            # Extract tonal features
            tonal_features_a = extract_tonal_features(audio_a)
            tonal_features_b = extract_tonal_features(audio_b)

            # Compute similarities
            tonal_similarity = np.dot(tonal_features_a, tonal_features_b) / (
                np.linalg.norm(tonal_features_a) * np.linalg.norm(tonal_features_b)
            )
            lyrics_similarity = compute_cosine_similarity(lyrics_a, lyrics_b) if not (
                is_instrumental_a or is_instrumental_b) else 0.1

            # Append features and label
            features.append(np.array([tonal_similarity, lyrics_similarity]))
            labels.append(label)

        logging.info("Feature extraction for pairs completed.")
        return np.array(features), np.array(labels)
    
    def calculate_song_features(self, audio):
        """
        Calculates lyrics and tonal features for a single song.
        """
        lyrics, is_instrumental = generate_lyrics(
            audio, instrumental_threshold=self.instrumental_threshold, model=self.lyrics_model
        )
        tonal_features = extract_tonal_features(audio)

        return lyrics, is_instrumental, tonal_features
    
    def calculate_similarity(self, tonal_features_a, tonal_features_b, lyrics_a, lyrics_b, is_instrumental_a, is_instrumental_b):
        """
        Calculates similarity for tonal features and lyrics.
        """
        tonal_similarity = np.dot(tonal_features_a, tonal_features_b) / (
            np.linalg.norm(tonal_features_a) * np.linalg.norm(tonal_features_b)
        )
        lyrics_similarity = compute_cosine_similarity(lyrics_a, lyrics_b) if not (
            is_instrumental_a or is_instrumental_b) else 0.1

        return tonal_similarity, lyrics_similarity
    
    def compute_similarity_and_predict(self, tonal_features_a, tonal_features_b, lyrics_a, lyrics_b, is_instrumental_a, is_instrumental_b):
        """
        Computes similarities and makes a prediction using the neural network model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model.to(device)
        if not self.is_model_loaded:
            raise RuntimeError("Model is not loaded. Call 'load_model()' first.")
        
        tonal_similarity, lyrics_similarity = self.calculate_similarity(
            tonal_features_a, tonal_features_b, lyrics_a, lyrics_b, is_instrumental_a, is_instrumental_b
        )

        features = np.array([[tonal_similarity, lyrics_similarity]])
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = self.nn_model(features_tensor).item()

        return prediction

    def train(self, X_train, y_train, num_epochs=10, learning_rate=0.001):
        """Train the classifier."""
        logging.info("Training classifier...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model = self.nn_model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)

        # Create DataLoader
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            self.nn_model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.nn_model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        logging.info("Training completed.")
        self.save_model("model.pth")

    def evaluate(self, X_test, y_test):
        """Evaluate the classifier."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model.eval()
        self.nn_model.to(device)

        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        labels = torch.tensor(y_test, dtype=torch.float32).to(device)

        accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
        precision = torchmetrics.classification.BinaryPrecision().to(device)
        recall = torchmetrics.classification.BinaryRecall().to(device)
        f1 = torchmetrics.classification.BinaryF1Score().to(device)

        with torch.no_grad():
            outputs = self.nn_model(inputs).squeeze()
            preds = (outputs > 0.5).float()

            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)

        logging.info(f"Accuracy: {accuracy.compute():.4f}")
        logging.info(f"Precision: {precision.compute():.4f}")
        logging.info(f"Recall: {recall.compute():.4f}")
        logging.info(f"F1 Score: {f1.compute():.4f}")

    def save_model(self, filepath):
        """Save the model to a file."""
        logging.info(f"Saving model to {filepath}...")
        torch.save(self.nn_model.state_dict(), filepath)
        logging.info("Model saved successfully.")

    
    def predict(self, audio_a, audio_b):
        """
        Predict whether a pair of audio files are covers of each other.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model.to(device)
        if not self.is_model_loaded:
            raise RuntimeError("Model is not loaded. Call 'load_model()' first.")
        
        logging.info("Preparing features for prediction...")
        # Extract features
        lyrics_a, is_instrumental_a = generate_lyrics(
            audio_a, instrumental_threshold=self.instrumental_threshold, model=self.lyrics_model
        )
        lyrics_b, is_instrumental_b = generate_lyrics(
            audio_b, instrumental_threshold=self.instrumental_threshold, model=self.lyrics_model
        )

        tonal_features_a = extract_tonal_features(audio_a)
        tonal_features_b = extract_tonal_features(audio_b)

        tonal_similarity = np.dot(tonal_features_a, tonal_features_b) / (
            np.linalg.norm(tonal_features_a) * np.linalg.norm(tonal_features_b)
        )
        lyrics_similarity = compute_cosine_similarity(lyrics_a, lyrics_b) if not (
            is_instrumental_a or is_instrumental_b) else 0.1

        # Combine features
        features = np.array([[tonal_similarity, lyrics_similarity]])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        features_tensor = features_tensor.to(device)

        # Predict
        with torch.no_grad():
            prediction = self.nn_model(features_tensor).item()

        logging.info(f"Prediction score (cover likelihood): {prediction:.4f}")
        return prediction

