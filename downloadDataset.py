import os
import json
import subprocess

OUTPUT_DIR = 'datasets/shs100k'

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load your JSON data
with open('datasets/shs100k.json', 'r') as f:
    data = json.load(f)

for entry in data:
    wav_path = entry['wav']  # e.g. datasets/shs100k/9FQXXlHTz1k.wav
    youtube_id = os.path.splitext(os.path.basename(wav_path))[0]
    youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
    output_wav = os.path.join(OUTPUT_DIR, f'{youtube_id}.wav')

    if os.path.exists(output_wav):
        print(f'Already downloaded: {output_wav}')
        continue

    print(f'Downloading {youtube_url} ...')

    # Download and convert to WAV using yt-dlp + ffmpeg
    try:
        subprocess.run([
            'yt-dlp',
            '-x', '--audio-format', 'wav',
            '-o', os.path.join(OUTPUT_DIR, f'{youtube_id}.%(ext)s'),
            youtube_url
        ], check=True)
        print(f'Downloaded: {output_wav}')
    except subprocess.CalledProcessError:
        print(f'Failed to download {youtube_url}')