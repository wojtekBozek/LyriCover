import json
from collections import Counter

# Load JSON data
with open("unique_songs.json", "r") as f:
    data = json.load(f)

# Count how many times each song appears
song_counts = Counter(entry["song"] for entry in data)

# Print each song and its count
for song, count in song_counts.items():
    print(f"{song} - {count}")

# Calculate statistics
total_songs = len(song_counts)
songs_with_covers = sum(1 for count in song_counts.values() if count > 1)
songs_without_covers = sum(1 for count in song_counts.values() if count == 1)
possible_combinations = sum(count * (count - 1) // 2 for count in song_counts.values() if count > 1)

# Print summary
print("\n--- Summary ---")
print(f"Total entries: {len(data)}")
print(f"Total unique songs: {total_songs}")
print(f"Songs with covers (multiple versions): {songs_with_covers}")
print(f"Songs without covers (single version): {songs_without_covers}")
print(f"Possible cover combinations: {possible_combinations}")
