import argparse
import logging
from model import CoverClassifier
from utils import read_metadata, load_whisper_model, generate_pairs, load_pairs, save_pairs
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
from sklearn.model_selection import train_test_split
import wandb
import os


def main_evaluate():
    '''Main function to run simple evaluation on the cover detection model.'''
    wandb.init(project="cover-detection-evaluation")
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("--metadata_path", type=str, default="datasets/metadata_augmented.json", help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--load_save", type=str, default="load", choices=["load", "save", None],
                        help="Specify whether to load or save lyrics embeddings.")
    parser.add_argument("--lyrics_dir", type=str, default="lyrics", help="Directory for storing/loading lyrics files.")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the model file.")
    parser.add_argument("--test_pairs_path", type=str, default="datasets/fixed_pairs/saved_test_pairs3.json", help="Path to the test pairs JSON file.")
    args = parser.parse_args()

    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)

    classifier = CoverClassifier(
        instrumental_threshold=args.instrumental_threshold,
        lyrics_model=whisper
    )
    if(os.path.exists(args.model_path)):
        classifier.load_model(args.model_path)

    if os.path.exists(args.test_pairs_path):
        test_pairs = load_pairs(args.test_pairs_path)
    else:
        test_pairs = generate_pairs(metadata, 4)
        save_pairs(test_pairs, args.test_pairs_path)

    features, labels = classifier.extract_pair_features(test_pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)

    classifier.evaluate(features, labels)
    logging.info("Evaluation completed successfully.")
    wandb.finish()  # Finish the WandB run


if __name__ == "__main__":
    main_evaluate()    