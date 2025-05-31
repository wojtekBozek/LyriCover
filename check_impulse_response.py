import os
import numpy as np
import soundfile as sf
from audiomentations import ApplyImpulseResponse
import warnings
import librosa

# Path to your impulse response directory
ir_dir = "selectedImpulseResponses"
dataset, sample_rate = librosa.load("audiomentations_augment.wav", sr=None)

# Generate a test sine wave (1-second duration)
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Loop through all impulse responses
problem_files = []

for filename in os.listdir(ir_dir):
    if not filename.lower().endswith((".wav", ".flac", ".aiff", ".aif")):
        continue

    ir_path = os.path.join(ir_dir, filename)
    
    try:
        augmenter = ApplyImpulseResponse(ir_path)
        augmented = augmenter(samples=test_audio, sample_rate=sample_rate)

        if not np.all(np.isfinite(augmented)) or np.linalg.norm(augmented) < 1e-5:
            print(f"Problem with IR: {filename} -> Invalid or silent result")
            problem_files.append(filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        problem_files.append(filename)

print("\nPossible corrupted or problematic IR files:")
for f in problem_files:
    print(f"- {f}")