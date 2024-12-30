import json
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import whisper
import torch
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Function to read the dataset metadata
def read_metadata(json_path):
    logging.info(f"Reading metadata from {json_path}...")
    try:
        data = json.load(open(json_path))
        logging.info(f"Successfully loaded metadata. Total records: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Failed to read metadata: {e}")
        raise

# Function to extract lyrics transcription using Whisper
def transcribe_audio(file_path, model):
    logging.info(f"Transcribing audio: {file_path}")
    try:
        result = model.transcribe(file_path)
        transcription = result["text"]
        logging.info(f"Transcription successful: {transcription[:30]}...")  # Log first 30 chars
        return transcription
    except Exception as e:
        logging.error(f"Error transcribing {file_path}: {e}")
        raise

# Function to extract tonal features (HPCP)
def extract_hpcp(file_path):
    logging.info(f"Extracting tonal features (HPCP) for: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        hpcp = np.mean(chroma, axis=1)
        logging.info("HPCP feature extraction successful.")
        return hpcp
    except Exception as e:
        logging.error(f"Error extracting HPCP features for {file_path}: {e}")
        raise

# Function to prepare dataset features and labels
def prepare_dataset(metadata, model):
    features = []
    labels = []

    for idx, item in enumerate(metadata):
        audio_path = item['wav']
        label = item['utt'].split('_')[0]  # Clique ID

        logging.info(f"Processing record {idx + 1}/{len(metadata)}: {audio_path}")
        try:
            transcription = transcribe_audio(audio_path, model)
            hpcp_features = extract_hpcp(audio_path)
            combined_features = np.concatenate((hpcp_features, np.array([len(transcription)])))
            features.append(combined_features)
            labels.append(label)
            logging.info(f"Record {idx + 1} processed successfully.")
        except Exception as e:
            logging.warning(f"Skipping record {idx + 1} due to error: {e}")

    logging.info(f"Dataset preparation complete. Total records processed: {len(features)}")
    return np.array(features), np.array(labels)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a classifier on audio data.")
    parser.add_argument("metadata_path", type=str, help="Path to the metadata JSON file.", default="output_reduced.json")
    args = parser.parse_args()

    # Paths and model setup
    metadata_path = args.metadata_path
    metadata = read_metadata(metadata_path)

    logging.info("Loading Whisper model...")
    model = whisper.load_model("large")  # You can choose "tiny", "base", "small", "medium", or "large"

    # Prepare dataset
    logging.info("Preparing dataset...")
    X, y = prepare_dataset(metadata, model)

    # Split into train and test sets
    logging.info("Splitting dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier
    logging.info("Training classifier...")
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    logging.info("Evaluating classifier...")
    y_pred = classifier.predict(X_test)
    logging.info("\n" + classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
