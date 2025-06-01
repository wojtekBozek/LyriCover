
from generate_lyrics import LyricsProcessor, process_folder    
import argparse
from utils import load_whisper_model


def main_generate_lyrics():
    '''Main function to generate lyrics for audio files in a specified folder.'''
    parser = argparse.ArgumentParser(description="Generate lyrics for audio files.")
    parser.add_argument("audio_folder", type=str, help="Path to the folder containing audio files.")
    parser.add_argument("--output_folder", type=str, default="lyrics", help="Output folder for generated lyrics.")
    parser.add_argument("--instrumental_threshold", type=float, default=10, help="Threshold for detecting instrumental songs.")
    args = parser.parse_args()

    # Initialize Whisper model
    whisper = load_whisper_model()

    # Initialize LyricsProcessor
    lyrics_processor = LyricsProcessor(whisper, args.instrumental_threshold)

    # Process the audio folder
    process_folder(args.audio_folder, lyrics_processor, args.output_folder)