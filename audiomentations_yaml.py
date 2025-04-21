import yaml
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, TimeMask
import librosa
import soundfile as sf

def load_augmentations_from_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    augmentation_list = []
    for aug in config["augmentations"]:
        aug_type = aug.pop("type")
        probability = aug.pop("probability", 1.0)
        augmentation_class = globals()[aug_type]
        augmentation_list.append(augmentation_class(p=probability, **aug))
    
    return Compose(augmentation_list)

if __name__ == "__main__":
    config_path = "augmentations.yaml"
    augment = load_augmentations_from_yaml(config_path)
    
    signal, sr = librosa.load("7JEjQG4-tpU.wav", sr=None)
    augmented_signal = augment(signal, sr)
    sf.write("audiomentations_augment.wav", augmented_signal, sr)