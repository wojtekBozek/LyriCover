from audiomentations import (
    AddGaussianNoise, PitchShift, HighPassFilter, TimeMask,
    ClippingDistortion, Shift, Gain, PolarityInversion, Compose
)
import random

ALL_AUGMENTATIONS = {
    "AddGaussianNoise": lambda: AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
    "PitchShift": lambda: PitchShift(min_semitones=-4, max_semitones=4, p=0.6),
    "HighPassFilter": lambda: HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=8000, p=0.6),
    "TimeMask": lambda: TimeMask(min_band_part=0.05, max_band_part=0.15, fade=True, p=0.6),
    "ClippingDistortion": lambda: ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=50, p=0.6),
    "Shift": lambda: Shift(min_fraction=-0.5, max_fraction=0.5, rollover=True, p=0.4),
    "Gain": lambda: Gain(min_gain_in_db=-12.0, max_gain_in_db=6.0, p=0.5),
    "PolarityInversion": lambda: PolarityInversion(p=0.3)
}


def generate_random_pipeline(n_augments):
    chosen = random.sample(list(ALL_AUGMENTATIONS.keys()), k=n_augments)
    return Compose([ALL_AUGMENTATIONS[name]() for name in chosen])