import argparse
import logging
import torch
from model import CoverClassifier
import shutil
from utils import read_metadata, load_whisper_model, generate_pairs, split_data, cover_stats, load_pairs, save_pairs
from audio_pair_dataset import AudioPairDataset
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
from sklearn.model_selection import train_test_split
from generate_lyrics import LyricsProcessor, process_folder
from augmentations_from_yaml import get_augmentation
from audiomentations_yaml import load_augmentations_from_yaml
import wandb
import os
import yaml

def main_train_from_loader():
    '''Main function to train with on the fly extracted features using audio pairs dataset and augmentations from a YAML configuration file.'''
    wandb.init(project="Lyricover-Augmentation-Compose") 
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("--metadata_path", type=str, default="datasets/shs100k_reduced_download_test.json", help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.5, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default="load", choices=["load", "save", None],
                        help="Specify whether to load or save lyrics embeddings.")
    parser.add_argument("--lyrics_dir", type=str, default="lyrics", help="Directory for storing/loading lyrics files.")
    parser.add_argument("--augmentation_type", type=str, default="augmentations.yaml", help="Path to YAML config for audio augmentations.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the model file.")
    parser.add_argument("--train_from_existing_pairs", action="store_true", default=True, help="Use existing pairs for training.")
    parser.add_argument("--save_generated_pairs", action="store_true", default=True, help="Save existing pairs if they do not exist.")
    parser.add_argument("--test_pairs_path", type=str, default="datasets/fixed_pairs/saved_test_pairs3.json", help="Path to the test pairs JSON file.")
    parser.add_argument("--train_pairs_path", type=str, default="datasets/fixed_pairs/saved_train_pairs3.json", help="Path to the train pairs JSON file.")
    parser.add_argument("--max_pairs", type=int, default=4, help="Maximum number of pairs to generate.")
    args = parser.parse_args()

    args_path = os.path.join(wandb.run.dir, "args.yaml")
    with open(args_path, "w") as f:
        yaml.dump(vars(args), f)
    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)

    classifier = CoverClassifier(
        instrumental_threshold=args.instrumental_threshold,
        lyrics_model=whisper
    )
    src_yaml_path = args.augmentation_type
    if not os.path.exists(src_yaml_path):
        logging.error(f"Augmentation YAML file {src_yaml_path} does not exist.")
        raise FileNotFoundError(f"Augmentation YAML file {src_yaml_path} does not exist.")
    aug_fn = load_augmentations_from_yaml(src_yaml_path)
    dst_yaml_path = os.path.join(wandb.run.dir, src_yaml_path)

    # Copy the file
    shutil.copy(src_yaml_path, dst_yaml_path)
    
    if args.train_from_existing_pairs and os.path.exists(args.train_pairs_path) and os.path.exists(args.test_pairs_path):
        logging.info("Loading existing pairs from disk...")
        train_pairs = load_pairs(args.train_pairs_path)
        test_pairs = load_pairs(args.test_pairs_path)
    else:
        logging.info("Generating new pairs of songs for cover classification...")
        pairs = generate_pairs(metadata, max_pairs=args.max_pairs)
        train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_split_size, random_state=42, shuffle=True)
        if(args.save_generated_pairs):
            save_pairs(train_pairs, args.train_pairs_path)
            save_pairs(test_pairs, args.test_pairs_path)
    train_dataset = AudioPairDataset(train_pairs, whisper, args.instrumental_threshold, augmentation_fn=aug_fn, lyrics_dir=args.lyrics_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    if(os.path.exists(args.model_path)):
        classifier.load_model(args.model_path)
    classifier.train_from_loader(train_loader, num_epochs=args.epochs, learning_rate=0.0001)

    model_save_path = os.path.join(wandb.run.dir, f"model.pth")
    classifier.save_model(model_save_path)
    features, labels = classifier.extract_pair_features(test_pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)

    classifier.evaluate(features, labels)
    wandb.finish()


if __name__ == "__main__":
    main_train_from_loader()
