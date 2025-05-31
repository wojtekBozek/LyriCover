import os
import shutil
import argparse
import soundfile as sf

def get_sample_rate(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return f.samplerate
    except RuntimeError as e:
        print(f"Error reading {file_path}: {e}")
        return None

def copy_matching_wavs_flat(src_dir, dst_dir, target_sr):
    os.makedirs(dst_dir, exist_ok=True)
    copied_count = 0

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                sr = get_sample_rate(full_path)
                if sr == target_sr:
                    dst_path = os.path.join(dst_dir, file)

                    # Handle duplicate filenames
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(dst_dir, f"{base}_{counter}{ext}")
                        counter += 1

                    shutil.copy2(full_path, dst_path)
                    print(f"Copied: {file}")
                    copied_count += 1
                else:
                    print(f"Skipped (SR {sr}): {file}")

    print(f"\nTotal files copied: {copied_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy WAVs with a given sample rate into a flat folder.")
    parser.add_argument("source", help="Source directory")
    parser.add_argument("destination", help="Destination directory")
    parser.add_argument("sample_rate", type=int, help="Target sample rate (e.g., 16000)")

    args = parser.parse_args()
    copy_matching_wavs_flat(args.source, args.destination, args.sample_rate)