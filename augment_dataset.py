import os
import shutil
import random
import yaml
import json
import numpy as np
import soundfile as sf
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

# ===== CONFIG =====
INPUT_YAML = os.path.join("datasets", "shs100k_reduced.json")  # or .json
AUDIO_BASE_DIR = "."  # Root where original .wav files are found
OUTPUT_DIR = os.path.join("datasets", "evaluation_augmented_set2")
AUGMENT_PROB = 0.5  # Probability to apply augmentation to a file
SAMPLE_RATE = 44100  # Change if your data is not 44.1kHz

# ===== AUGMENTATION PIPELINE =====
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5)
])

# ===== LOAD INPUT METADATA =====
def load_metadata(path):
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported file format")

# ===== MAIN PROCESSING =====
def process_files(metadata):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_metadata = []

    for entry in metadata:
        utt_id = entry["utt"]
        input_path = os.path.join(AUDIO_BASE_DIR, entry["wav"])
        filename = os.path.basename(input_path)
        base_name, ext = os.path.splitext(filename)

        # Decide whether to augment
        should_augment = random.random() < AUGMENT_PROB

        # Set output filename
        if should_augment:
            output_filename = f"{base_name}_aug{ext}"
        else:
            output_filename = f"{base_name}{ext}"

        output_path = os.path.join(OUTPUT_DIR, output_filename)

        try:
            audio, sr = librosa.load(input_path, sr=None)
            if sr != SAMPLE_RATE:
                print(f"WARNING: {filename} has sample rate {sr}, expected {SAMPLE_RATE}")

            if should_augment:
                audio = augment(samples=audio, sample_rate=sr)

            sf.write(output_path, audio, sr)

            new_entry = entry.copy()
            new_entry["wav"] = output_path
            new_entry["utt"] = f"{utt_id}_aug" if should_augment else utt_id
            result_metadata.append(new_entry)

            print(f"✓ Processed: {filename} {'[AUG]' if should_augment else '[COPY]'}")

        except Exception as e:
            print(f"✗ Failed: {filename} — {e}")

    return result_metadata

# ===== EXECUTION =====
if __name__ == "__main__":
    print("🔄 Loading metadata...")
    metadata = load_metadata(INPUT_YAML)

    print("🎧 Processing files...")
    new_metadata = process_files(metadata)

    # Save new metadata
    output_meta_path = os.path.join("datasets", "metadata_augmented2.json")
    with open(output_meta_path, "w") as f:
        json.dump(new_metadata, f , indent=2)

    print(f"✅ Done. Augmented files and metadata saved to: {OUTPUT_DIR}")
