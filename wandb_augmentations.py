import yaml
import wandb
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, TimeMask, ClippingDistortion 
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
    
    return Compose(augmentation_list), config

if __name__ == "__main__":
    # Initialize W&B
    wandb.init(project="audio-augmentations", name="augmentation-run")

    config_path = "augmentations.yaml"
    augment, config = load_augmentations_from_yaml(config_path)

    # Log the YAML configuration to W&B
    wandb.config.update(config)

    # Load and augment the audio signal
    signal, sr = librosa.load("examples/bleach.wav", sr=None)
    augmented_signal = augment(signal, sr)

    # Save the augmented audio
    output_path = "audiomentations_augment.wav"
    sf.write(output_path, augmented_signal, sr)

    # Log the augmented audio as an artifact
    wandb.log({"sample_rate": sr})
    #wandb.log({"original_audio": wandb.Audio("examples/bleach.wav", sample_rate=sr)})
    #wandb.log({"augmented_audio": wandb.Audio(output_path, sample_rate=sr)})

    # Finish the W&B run
    wandb.finish()