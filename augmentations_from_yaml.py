from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, ClippingDistortion, HighPassFilter, TimeMask, Mp3Compression, PolarityInversion, Gain, GainTransition, LowPassFilter

def get_augmentation(cfg):
    aug_type = cfg['augmentation_type']

    if aug_type == "none":
        return None
    elif aug_type == "pitch_shift_1":
        return Compose([
            PitchShift(min_semitones=-4, max_semitones=4, p=0.8)
        ])
    elif aug_type == "time_stretch_1":
        return Compose([
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.8)
        ])    
    elif aug_type == "GaussianNoise_1":
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05, p=0.8)
        ])
    elif aug_type == "clipping_distortion_1":
        return Compose([
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.8)
        ])
    elif aug_type == "clipping_distortion_2":
        return Compose([
            ClippingDistortion(min_percentile_threshold=30, max_percentile_threshold=60, p=0.8)
        ])
    elif aug_type == "combined_gain":
        return Compose([
            Gain(min_gain_db=-10, max_gain_db=10, p=0.8)
        ])
    elif aug_type == "negative_gain":
        return Compose([
            Gain(min_gain_db=-10, max_gain_db=0, p=0.8)
        ])
    elif aug_type == "positive_gain":
        return Compose([
            Gain(min_gain_db=0, max_gain_db=10, p=0.8)
        ])
    elif aug_type == "transition_gain":
        return Compose([
            GainTransition(min_gain_db=-10, max_gain_db=10, p=0.8)
        ])
    elif aug_type == "low_pass_filter":
        return Compose([
            LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=8000, p=0.8)
        ])
    elif aug_type == "high_pass_filter":
        return Compose([
            HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=500, p=0.8)
        ])
    elif aug_type == "Mp3Compression_1":
        return Compose([
            Mp3Compression(min_bitrate= 64, max_bitrate=128, p=0.8)
        ])
    elif aug_type == "Mp3Compression_2":
        return Compose([
            Mp3Compression(min_bitrate= 8, max_bitrate=32, p=0.8)
        ])
    elif aug_type == "polarity_inversion":
        return Compose([
            PolarityInversion(p=0.8)
        ])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")