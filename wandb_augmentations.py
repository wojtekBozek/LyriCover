import wandb
import librosa
import soundfile as sf
from augmentations_from_yaml import get_augmentation

AUGMENTATIONS = [
    "none",
    "pitch_shift_1",
    "pitch_shift_2",
    "time_stretch_1",
    "time_stretch_2",
    "GaussianNoise_1",
    "GaussianNoise_2",
    "clipping_distortion_1",
    "clipping_distortion_2",
    "high_pass_filter",
    "low_pass_filter",
    "Mp3Compression_1",
    "Mp3Compression_2",
    "negative_gain",
    "positive_gain",
    "transition_gain",
    "polarity_inversion"
]

AUDIO_PATH = "examples/bleach.wav"

for aug_name in AUGMENTATIONS:
    wandb.init(project="audio-augmentations", name=f"test_{aug_name}")

    config = {"augmentation_type": aug_name}
    wandb.config.update(config)

    # Load augmentation and audio
    augment = get_augmentation(config)
    signal, sr = librosa.load(AUDIO_PATH, sr=None)

    # Apply augmentation
    if augment:
        augmented = augment(signal, sr)
    else:
        augmented = signal

    # Save and log
    output_path = f"output_augmented_{aug_name}.wav"
    sf.write(output_path, augmented, sr)

    wandb.log({
        "sample_rate": sr,
        #"original_audio": wandb.Audio(AUDIO_PATH, sample_rate=sr),
        #"augmented_audio": wandb.Audio(output_path, sample_rate=sr),
    })

    wandb.finish()