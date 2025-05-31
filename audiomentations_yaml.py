import yaml
from audiomentations import Compose, Shift, AddGaussianNoise, PitchShift, HighPassFilter, TimeMask, ApplyImpulseResponse, TimeStretch, ClippingDistortion, Mp3Compression, PolarityInversion, Gain, GainTransition, LowPassFilter
import librosa
import os
import soundfile as sf
import logging


def load_augmentations_from_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    augmentation_list = []
    for aug in config["augmentations"]:
        aug_type = aug.pop("type")
        probability = aug.pop("probability", 1.0)

        if aug_type == "ApplyImpulseResponse":
            ir_path = aug.get("ir_path")
            if ir_path and os.path.isdir(ir_path) and any(f.endswith((".wav", ".flac")) for f in os.listdir(ir_path)):
                augmentation = ApplyImpulseResponse(ir_path=ir_path, p=probability)
                augmentation_list.append(augmentation)
            else:
                logging.warning(f"Skipping ApplyImpulseResponse: directory '{ir_path}' is missing or empty.")
        else:
            augmentation_class = globals()[aug_type]
            augmentation_list.append(augmentation_class(p=probability, **aug))

    return Compose(augmentation_list)

if __name__ == "__main__":
    config_path = "augmentations.yaml"
    augment = load_augmentations_from_yaml(config_path)
    
    signal, sr = librosa.load("examples/bleach.wav", sr=None)
    augmented_signal = augment(signal, sr)
    sf.write("audiomentations_augment.wav", augmented_signal, sr)