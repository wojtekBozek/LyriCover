from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter


from audiomentations import TimeMask
import numpy as np
import librosa
import soundfile as sf


augment= Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
    PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
    HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=8000, p=1.0),
    # BandPassFilter(min_center_freq=2000, max_center_freq=8000, p=1.0),
    # ApplyImpulseResponse(ir_path="/path/to/sound_folder", p=1.0),
    TimeMask(min_band_part=0.05, max_band_part=0.15, fade=True, p=1.0)
])


if __name__ == "__main__":
    signal, sr = librosa.load("examples/bleach.wav", sr=None)
    augmented_signal = augment(signal, sr)
    sf.write("audiomentations_augment.wav", augmented_signal, sr)