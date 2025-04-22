import os

# Change this to your target directory
folder_path = "lyrics"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        base = filename[:-4]  # remove ".txt"
        new_filename = f"{base}.wav.txt"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")