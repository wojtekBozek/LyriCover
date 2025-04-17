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
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.2, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default=None, choices=["load", "save", None],
                        help="Specify whether to load or save lyrics embeddings.")
    parser.add_argument("--lyrics_dir", type=str, default="lyrics", help="Directory for storing/loading lyrics files.")
    parser.add_argument("--augmentation_config", type=str, default=None, help="Path to YAML config for audio augmentations.")
    args = parser.parse_args()

    # Initialize Whisper model and read metadata
    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)
    pairs = generate_pairs(metadata)

    # Initialize the classifier with Whisper model and instrumental threshold
    classifier = CoverClassifier(
        instrumental_threshold=args.instrumental_threshold,
        lyrics_model=whisper,
        augmentation_config=args.augmentation_config
    )

    # Extract features with specified load_save option and lyrics directory
    features, labels = classifier.extract_pair_features(pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)
    X_train, X_test, y_train, y_test = split_data(features, labels, args.test_split_size)

    # Display cover stats and train the classifier
    cover_stats(X_train, y_train, X_test, y_test)
    classifier.train(X_train, y_train)

    # Evaluate the trained model
    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
