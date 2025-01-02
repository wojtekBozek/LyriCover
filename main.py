import argparse
import logging
import torch
from model import CoverClassifier
from utils import read_metadata, load_whisper_model, generate_pairs, split_data, cover_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("metadata_path", type=str, help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=8, help="Threshold for detecting instrumental songs.")
    args = parser.parse_args()

    # Initialize Whisper model and read metadata
    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)
    pairs = generate_pairs(metadata)

    classifier = CoverClassifier(instrumental_threshold=args.instrumental_threshold, lyrics_model= whisper)

    features, labels = classifier.extract_pair_features(pairs)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    cover_stats(X_train, y_train, X_test, y_test)

    classifier.train(X_train, y_train)

    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
