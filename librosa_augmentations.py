import librosa
import numpy as np
import soundfile as sf

def add_white_noise(audio, noise_factor=0.005):
    """
    Add white noise to the audio signal.
    
    Parameters:
    - audio: numpy array, the original audio signal
    - noise_factor: float, the factor by which to scale the noise
    
    Returns:
    - numpy array, the audio signal with added noise
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio


def add_environmental_noise(audio, noise_factor=0.005):
    """
    Add environmental noise to the audio signal.
    
    Parameters:
    - audio: numpy array, the original audio signal
    - noise_factor: float, the factor by which to scale the noise
    
    Returns:
    - numpy array, the audio signal with added noise
    """
    # Load environmental noise (e.g., white noise)
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def pitch_augmentation(audio, num_semitones=2, sr=22050):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=num_semitones)

def time_stretch_augmentation(signal, stretch_rate=1.0):
    return librosa.effects.time_stretch(signal, rate=stretch_rate)

def noise_injection_augmentation():
    """
    Apply noise injection augmentation to the audio data.
    """
    pass

def random_crop_augmentation():
    """
    Apply random cropping augmentation to the audio data.
    """
    pass


def add_impulse_respone():
    """
    Apply impulse response augmentation to the audio data.
    """
    pass


def random_gain_augmentation(audio, gain_range=(-10, 10)):
    """
    Apply random gain augmentation to the audio data.
    
    Parameters:
    - audio: numpy array, the original audio signal
    - gain_range: tuple, the range of gain in dB
    
    Returns:
    - numpy array, the audio signal with applied gain
    """
    gain = np.random.uniform(gain_range[0], gain_range[1])
    augmented_audio = audio * (10 ** (gain / 20))
    return augmented_audio


def invert_polarity(audio):
    """
    Apply invert polarity augmentation to the audio data.
    """
    return audio * -1


if __name__ == "__main__":
    # Example usage
    audio_path = "examples/bleach.wav"
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Apply white noise augmentation
    #augmented_audio = add_white_noise(audio, noise_factor=0.005)
    #augmented_audio = time_stretch_augmentation(augmented_audio, stretch_rate=0.8)
    #augmented_audio = pitch_augmentation(audio, num_semitones=2, sr=sr)
    # Save the augmented audio
    augmented_audio = invert_polarity(audio)
    #augmented_audio = random_gain_augmentation(augmented_audio, gain_range=(-10, 10))
    sf.write("augmented_audio.wav", augmented_audio, sr)  # Using soundfile to save the audio.