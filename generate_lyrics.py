import os
import logging

from text_utils import generate_lyrics


class LyricsProcessor:
    def __init__(self, lyrics_model, instrumental_threshold=0.5):
        self.lyrics_model = lyrics_model
        self.instrumental_threshold = instrumental_threshold

    def load_lyrics(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                return file.read()
        return None

    def save_lyrics(self, filepath, lyrics):
        with open(filepath, 'w') as file:
            file.write(lyrics)

    def is_empty_or_stop_words(self, lyrics):
        # Replace with your actual logic to check for empty or stopword-only lyrics
        return len(lyrics.strip()) == 0

    def get_lyrics(self, audio_path, filepath, load_save=None):
        lyrics = self.load_lyrics(filepath) if load_save == "load" else None
        if lyrics is None:
            logging.info(f"Lyrics not found for {audio_path}, generating...")
            lyrics, is_instrumental = generate_lyrics(
                audio_path, instrumental_threshold=self.instrumental_threshold, model=self.lyrics_model
            )
            if not lyrics or self.is_empty_or_stop_words(lyrics):
                logging.warning(f"Lyrics for {audio_path} are empty or only stop words. Marked instrumental.")
                lyrics, is_instrumental = "", True  # ⬅️ empty string to save blank .txt
            if load_save == "save":
                self.save_lyrics(filepath, lyrics)
        else:
            is_instrumental = (lyrics.strip() == "")
        return lyrics, is_instrumental

def process_folder(folder_path, lyrics_processor, output_folder=None):
    if output_folder is None:
        output_folder = folder_path

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.wav'):
            audio_path = os.path.join(folder_path, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)

            lyrics, is_instrumental = lyrics_processor.get_lyrics(audio_path, txt_path, load_save="save")
            if lyrics:
                print(f"Lyrics saved for {filename}")
            elif is_instrumental:
                print(f"{filename} is instrumental. No lyrics saved.")
            else:
                print(f"Failed to generate lyrics for {filename}")

# Example usage:
# Make sure you define or import `generate_lyrics` and provide a valid model
# lyrics_model = ...  # Your lyrics generation model
# processor = LyricsProcessor(lyrics_model)
# process_folder('/path/to/your/wav/files', processor)
