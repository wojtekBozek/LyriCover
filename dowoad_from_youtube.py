import os
import subprocess

youtube_id = "7JEjQG4-tpU"

youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
output_wav = os.path.join("./", f'{youtube_id}.wav')

if os.path.exists(output_wav):
    print(f'Already downloaded: {output_wav}')
    
else: 
    print(f'Downloading {youtube_url} ...')
    # Download and convert to WAV using yt-dlp + ffmpeg
    try:
        subprocess.run([
            'yt-dlp',
            '-x', '--audio-format', 'wav',
            '-o', os.path.join("./", f'{youtube_id}.%(ext)s'),
            youtube_url
        ], check=True)
        print(f'Downloaded: {output_wav}')
    except subprocess.CalledProcessError:
        print(f'Failed to download {youtube_url}')