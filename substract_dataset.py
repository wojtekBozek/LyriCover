import json
from collections import defaultdict
import random

# Load the full dataset
with open("datasets/shs100k.json", "r") as f:
    data = json.load(f)

# Organize entries by song
song_dict = defaultdict(list)
for entry in data:
    song_dict[entry["song"]].append(entry)

# Split songs into covers and non-covers
cover_songs = [v for v in song_dict.values() if len(v) > 1]
non_cover_songs = [v for v in song_dict.values() if len(v) == 1]

# Shuffle for randomness
random.shuffle(cover_songs)
random.shuffle(non_cover_songs)

# Pick 1000 songs with covers (all versions) and 1000 non-covers
selected_cover_entries = [entry for group in cover_songs[:1000] for entry in group]
selected_non_cover_entries = [group[0] for group in non_cover_songs[:100]]

# Combine and save
final_subset = selected_cover_entries + selected_non_cover_entries

# Save to a new JSON file
with open("subset_1000_covers_1000_non.json", "w") as f:
    json.dump(final_subset, f, indent=4)

print(f"Saved {len(final_subset)} entries to 'subset_1000_covers_1000_non.json'")