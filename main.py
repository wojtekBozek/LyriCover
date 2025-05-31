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

def main_train_augment():
    wandb.init(project="cover-detection-singular_augmentation-test")
    config = wandb.config  
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("--metadata_path", type=str, default="datasets/shs100k_unique.json", help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.2, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default="load", choices=["load", "save", None],
                        help="Specify whether to load or save lyrics embeddings.")
    parser.add_argument("--lyrics_dir", type=str, default="lyrics", help="Directory for storing/loading lyrics files.")
    parser.add_argument("--augmentation_type", type=str, default=None, help="Path to YAML config for audio augmentations.")
    
    args = parser.parse_args()

    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)

    classifier = CoverClassifier(
        instrumental_threshold=args.instrumental_threshold,
        lyrics_model=whisper
    )

    aug_fn = get_augmentation({"augmentation_type": config.augmentation_type})
    wandb.log({"augmentation_type": config.augmentation_type})
    classifier.load_model()

    if os.path.exists("datasets/fixed_pairs/saved_train_pairs2.json"):
        train_pairs = load_pairs("datasets/fixed_pairs/saved_train_pairs2")
        test_pairs = load_pairs("datasets/fixed_pairs/saved_test_pairs2")
    else:
        pairs = generate_pairs(metadata)
        train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_split_size, random_state=42, shuffle=True)
        save_pairs(train_pairs, "datasets/fixed_pairs/saved_train_pairs2")
        save_pairs(test_pairs, "datasets/fixed_pairs/saved_test_pairs2")
    train_dataset = AudioPairDataset(train_pairs, whisper, args.instrumental_threshold, augmentation_fn=aug_fn, lyrics_dir=args.lyrics_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    classifier.train_from_loader(train_loader, num_epochs=3, learning_rate=0.001)

    model_save_path = os.path.join(wandb.run.dir, f"{config.augmentation_type}_model.pth")
    classifier.save_model(model_save_path)
    features, labels = classifier.extract_pair_features(test_pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)

    classifier.evaluate(features, labels)


def main_train_augmentCompose():
    wandb.init(project="cover-detection")
    config = wandb.config  
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("--metadata_path", type=str, default="datasets/shs100k_unique.json", help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.2, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default="load", choices=["load", "save", None],
                        help="Specify whether to load or save lyrics embeddings.")
    parser.add_argument("--lyrics_dir", type=str, default="lyrics", help="Directory for storing/loading lyrics files.")
    parser.add_argument("--augmentation_type", type=str, default=None, help="Path to YAML config for audio augmentations.")
    
    args = parser.parse_args()

    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)

    classifier = CoverClassifier(
        instrumental_threshold=args.instrumental_threshold,
        lyrics_model=whisper
    )


    src_yaml_path = "augmentations.yaml"
    aug_fn = load_augmentations_from_yaml(src_yaml_path)
    dst_yaml_path = os.path.join(wandb.run.dir, src_yaml_path)

    # Copy the file
    shutil.copy(src_yaml_path, dst_yaml_path)
    
    if os.path.exists("datasets/fixed_pairs/saved_train_pairs2.json"):
        train_pairs = load_pairs("datasets/fixed_pairs/saved_train_pairs2")
        test_pairs = load_pairs("datasets/fixed_pairs/saved_test_pairs2")
    else:
        pairs = generate_pairs(metadata)
        train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_split_size, random_state=42, shuffle=True)
        save_pairs(train_pairs, "datasets/fixed_pairs/saved_train_pairs2")
        save_pairs(test_pairs, "datasets/fixed_pairs/saved_test_pairs2")
    train_dataset = AudioPairDataset(train_pairs, whisper, args.instrumental_threshold, augmentation_fn=aug_fn, lyrics_dir=args.lyrics_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    classifier.load_model(os.path.join("wandb", "run-20250526_165234-y91r1dij", "files", "model.pth"))
    #classifier.load_model("model.pth")
    classifier.train_from_loader(train_loader, num_epochs=1, learning_rate=0.0001)

    model_save_path = os.path.join(wandb.run.dir, f"model.pth")
    classifier.save_model(model_save_path)
    features, labels = classifier.extract_pair_features(test_pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)

    classifier.evaluate(features, labels)

def main_evaluate():
    wandb.init(project="cover-detection-evaluate-augmented_dataset")
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("--metadata_path", type=str, default="datasets/metadata_augmented.json", help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.99, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default="load", choices=["load", "save", None],
                        help="Specify whether to load or save lyrics embeddings.")
    parser.add_argument("--lyrics_dir", type=str, default="lyrics", help="Directory for storing/loading lyrics files.")
    parser.add_argument("--augmentation_type", type=str, default=None, help="Path to YAML config for audio augmentations.")
    
    args = parser.parse_args()

    whisper = load_whisper_model()
    metadata = read_metadata(args.metadata_path)

    classifier = CoverClassifier(
        instrumental_threshold=args.instrumental_threshold,
        lyrics_model=whisper
    )
    classifier.load_model(os.path.join("wandb", "run-20250501_021717-qw00sbd5", "files", "time_stretch_1_model.pth"))
    #classifier.load_model(os.path.join("wandb", "run-20250526_165234-y91r1dij", "files", "model.pth"))
    #classifier.load_model("model.pth")

    if os.path.exists("saved_test_pairs_augmented.json"):
        #train_pairs = load_pairs("saved_train_pairs_augmented")
        test_pairs = load_pairs("saved_test_pairs_augmented")
    else:
        pairs = generate_pairs(metadata, 200)
        train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_split_size, random_state=42, shuffle=True)
        #save_pairs(train_pairs, "saved_train_pairs_augmented")
        save_pairs(test_pairs, "saved_test_pairs_augmented")

    features, labels = classifier.extract_pair_features(test_pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)

    classifier.evaluate(features, labels)
    

def main_generate_lyrics():
    parser = argparse.ArgumentParser(description="Generate lyrics for audio files.")
    parser.add_argument("audio_folder", type=str, help="Path to the folder containing audio files.")
    parser.add_argument("--output_folder", type=str, default="lyrics", help="Output folder for generated lyrics.")
    parser.add_argument("--instrumental_threshold", type=float, default=10, help="Threshold for detecting instrumental songs.")
    args = parser.parse_args()

    # Initialize Whisper model
    whisper = load_whisper_model()

    # Initialize LyricsProcessor
    lyrics_processor = LyricsProcessor(whisper, args.instrumental_threshold)

    # Process the audio folder
    process_folder(args.audio_folder, lyrics_processor, args.output_folder)


def main2():
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("metadata_path", type=str, help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.2, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default="save", choices=["load", "save", None],
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
    classifier.load_model()
    # Extract features with specified load_save option and lyrics directory
    train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_split_size, random_state=42, shuffle=True)

    train_dataset = AudioPairDataset(train_pairs, whisper, args.instrumental_threshold, lyrics_dir=args.lyrics_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Use this loader in classifier.train()
    classifier.train_from_loader(train_loader)

    features, labels = classifier.extract_pair_features(test_pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)
    # Evaluate the trained model
    classifier.evaluate(features, labels)


def main():
    parser = argparse.ArgumentParser(description="Pair-based cover detection system with PyTorch classifier.")
    parser.add_argument("metadata_path", type=str, help="Path to the metadata JSON file.")
    parser.add_argument("--instrumental_threshold", type=int, default=10, help="Threshold for detecting instrumental songs.")
    parser.add_argument("--test_split_size", default=0.2, type=float, help="Test size percentage for training and validation.")
    parser.add_argument("--load_save", type=str, default="save", choices=["load", "save", None],
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
    classifier.load_model()
    # Extract features with specified load_save option and lyrics directory
    features, labels = classifier.extract_pair_features(pairs, load_save=args.load_save, lyrics_dir=args.lyrics_dir)
    X_train, X_test, y_train, y_test = split_data(features, labels, args.test_split_size)

    # Display cover stats and train the classifier
    cover_stats(X_train, y_train, X_test, y_test)
    classifier.train(X_train, y_train)

    # Evaluate the trained model
    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main_train_augmentCompose()
