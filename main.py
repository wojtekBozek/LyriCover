import json
import librosa
import numpy as np
import whisper
import argparse
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
import random
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchmetrics
import string


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Parameters
SIMILARITY_THRESHOLD = 0.5  # Adjust as needed
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8  # 80% for training, 20% for testing
BALANCE_RATIO = 1  # Ratio of cover pairs to non-cover pairs in both train and test sets (1 means balanced)
MAX_PAIRS = 1000 # Maximum total number of pairs to generate (cover + non-cover)
INSTRUMENTAL_THRESHOLD = 8  # Number of words below which a song is considered instrumental

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Function to load Whisper model
def load_whisper_model():
    logging.info("Loading Whisper model...")
    return whisper.load_model("small")

# Function to read metadata
def read_metadata(json_path):
    logging.info(f"Reading metadata from {json_path}...")
    try:
        data = json.load(open(json_path))
        logging.info(f"Successfully loaded metadata. Total records: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Failed to read metadata: {e}")
        raise

# Function to transcribe audio
def transcribe_audio(file_path, model):
    logging.info(f"Transcribing audio for {file_path}...")
    try:
        result = model.transcribe(file_path)
        transcription = result['text'].lower()  # Convert transcription to lowercase
        transcription = transcription.replace(",", "")  # Remove commas
        transcription = transcription.replace(".", "")  # Remove dots (periods)
        transcription = ' '.join(transcription.split())  # Remove extra spaces
        logging.info(f"Transcription successful: {transcription[:30]}...")  # Log first 30 chars
        return transcription
    except Exception as e:
        logging.error(f"Error transcribing {file_path}: {e}")
        return ""

# Function to extract tonal features (HPCP)
def extract_tonal_features(file_path):
    logging.info(f"Extracting tonal features for: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        tonal_features = np.mean(chroma, axis=1)  # Mean across frames
        logging.info("Tonal feature extraction successful.")
        return tonal_features
    except Exception as e:
        logging.error(f"Error extracting tonal features for {file_path}: {e}")
        return np.zeros(12)

# Function to simulate lyrics generation and detect instrumental songs
def generate_lyrics(file_path, model):
    logging.info(f"Generating lyrics for: {file_path}")
    try:
        transcription = transcribe_audio(file_path, model)
        
        # Remove unnecessary spaces and punctuation
        transcription = transcription.strip().lower()

        # Check if song is instrumental (based on word count)
        word_count = len(transcription.split())
        is_instrumental = word_count < INSTRUMENTAL_THRESHOLD

        # Simulate additional processing or augmentation for generated lyrics
        lyrics = transcription.replace(".", " ").replace(",", " ").strip()
        
        logging.info(f"Lyrics generated: {lyrics[:30]}... (Instrumental: {is_instrumental})")  # Log first 30 chars
        return lyrics, is_instrumental
    except Exception as e:
        logging.error(f"Error generating lyrics for {file_path}: {e}")
        return "", False  # Default to not instrumental if there's an error

import random
from itertools import combinations
from collections import defaultdict

import random
from itertools import combinations
from collections import defaultdict

import random
from itertools import combinations
from collections import defaultdict

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
    return cover_pairs, not_cover_pairs


# Preprocessing text and calculating 3-gram TF-IDF
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Function to compute cosine similarity using 3-gram TF-IDF
def compute_cosine_similarity(text1, text2):
    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Convert texts to 3-gram TF-IDF vectors
    vectorizer = TfidfVectorizer(ngram_range=(3, 3))  # Use only 3-grams
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

def extract_pair_features(pairs, model):
    logging.info("Extracting features for pairs...")
    features = []
    labels = []

    for pair in pairs:
        song_a, song_b, label = pair

        # Extract features for each song in the pair
        audio_a, audio_b = song_a['wav'], song_b['wav']
        lyrics_a, is_instrumental_a = generate_lyrics(audio_a, model)
        lyrics_b, is_instrumental_b = generate_lyrics(audio_b, model)
        tonal_features_a = extract_tonal_features(audio_a)
        tonal_features_b = extract_tonal_features(audio_b)

        # Compute tonal similarity
        tonal_similarity = np.dot(tonal_features_a, tonal_features_b) / (
            np.linalg.norm(tonal_features_a) * np.linalg.norm(tonal_features_b)
        )

        # Compute lyrics similarity (using cosine similarity)
        try:
            lyrics_similarity = compute_cosine_similarity(lyrics_a, lyrics_b)
        except ValueError:
            is_instrumental_a = True
            is_instrumental_b = True
        
        # Neutralize lyrics similarity for instrumental songs
        if is_instrumental_a or is_instrumental_b:
            lyrics_similarity /= 10  # Neutralize lyrics similarity for instrumental songs
        # Append the features (tonal_similarity and lyrics_similarity) as a 2D vector
        features.append(np.array([tonal_similarity, lyrics_similarity]))  # Use two features
        labels.append(label)

    logging.info("Feature extraction for pairs completed.")
    return np.array(features), np.array(labels)

# Split data into train and test sets
def split_data(features, labels):
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_SEED)
    logging.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

# Define PyTorch Classifier
class CoverClassifier(nn.Module):
    def __init__(self):
        super(CoverClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2 input features (similarity + instrumental flag)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

def train_classifier(model, train_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = (outputs.squeeze() > 0.5).long()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    save_model(model, 'model.pth')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def custom_metrics(y_true, y_pred):
    # Convert inputs to NumPy arrays if they're tensors
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision, recall, and F1 score calculation with handling of zero division
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    
    return precision, recall, f1


def count_cover_not_cover_pairs(X_train, y_train, X_test, y_test):
    # Count the number of cover (1) and not-cover (0) pairs in the train set
    train_cover_count = np.sum(y_train == 1)
    train_not_cover_count = np.sum(y_train == 0)

    # Count the number of cover (1) and not-cover (0) pairs in the test set
    test_cover_count = np.sum(y_test == 1)
    test_not_cover_count = np.sum(y_test == 0)

    # Print the results
    logging.info(f"Train set - Cover pairs: {train_cover_count}, Not-cover pairs: {train_not_cover_count}")
    logging.info(f"Test set - Cover pairs: {test_cover_count}, Not-cover pairs: {test_not_cover_count}")
    
    return train_cover_count, train_not_cover_count, test_cover_count, test_not_cover_count

def evaluate_classifier(model, X_test, y_test, device="cuda"):
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode
    
    # Move the data to the device
    inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
    labels = torch.tensor(y_test, dtype=torch.long).to(device)
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(inputs)
        
        # Binary predictions (0 or 1)
        preds = (outputs.squeeze() > 0.5).long()
        
        # Move the predictions and labels back to the CPU for custom calculations
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()

        # Calculate custom metrics
        precision, recall, f1 = custom_metrics(labels_cpu, preds_cpu)

        # Log the results
        logging.info(f"Accuracy: {accuracy_score(labels_cpu, preds_cpu):.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")


def evaluate_classifier_with_torchmetrics(model, X_test, y_test, device):
    # Move data to the correct device
    model.eval()
    inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
    labels = torch.tensor(y_test, dtype=torch.long).to(device)

    # Use torchmetrics for metrics calculation
      # Use torchmetrics for metrics calculation
    accuracy = torchmetrics.classification.Accuracy(task='binary').to(device)  # specify task
    precision = torchmetrics.classification.Precision(num_classes=2, average='binary', task='binary').to(device)
    recall = torchmetrics.classification.Recall(num_classes=2, average='binary', task='binary').to(device)
    f1 = torchmetrics.classification.F1Score(num_classes=2, average='binary', task='binary').to(device)
    # Make predictions
    with torch.no_grad():
        outputs = model(inputs)
        preds = (outputs.squeeze() > 0.5).long()  # Assuming a binary classification

    # Update metrics
    accuracy.update(preds, labels)
    precision.update(preds, labels)
    recall.update(preds, labels)
    f1.update(preds, labels)

    # Compute metrics
    accuracy_result = accuracy.compute()
    precision_result = precision.compute()
    recall_result = recall.compute()
    f1_result = f1.compute()

    # Log results
    logging.info(f"Accuracy: {accuracy_result:.4f}")
    logging.info(f"Precision: {precision_result:.4f}")
    logging.info(f"Recall: {recall_result:.4f}")
    logging.info(f"F1 Score: {f1_result:.4f}")

    # Clear metric states for next evaluation
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()

    return accuracy_result, precision_result, recall_result, f1_result


def save_model(model, filepath):
    logging.info(f"Saving model to {filepath}...")
    torch.save(model.state_dict(), filepath)
    logging.info("Model saved successfully.")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("metadata_path", type=str, help="Path to the metadata JSON file.")
    args = parser.parse_args()

    metadata = read_metadata(args.metadata_path)
    model = load_whisper_model()
    device = torch.device("cuda")
    # Generate pairs and extract features
    cover_pairs, not_cover_pairs = generate_pairs(metadata)
    pairs = cover_pairs + not_cover_pairs
    features, labels = extract_pair_features(pairs, model)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(features, labels)
    train_cover_count, train_not_cover_count, test_cover_count, test_not_cover_count = count_cover_not_cover_pairs(X_train, y_train, X_test, y_test)
    # Define and train the model
    classifier = CoverClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Convert data to DataLoader format for PyTorch
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # Train the classifier
    train_classifier(classifier, train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate on the test set
    evaluate_classifier_with_torchmetrics(classifier, X_test, y_test, device=device)

if __name__ == "__main__":
    main()
