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

# Function to create pairs from a list of songs (covers or non-covers)
def create_pairs(songs):
    pairs = []
    for song_group in songs:
        if len(song_group) > 1:
            # Pair the songs within the group (covers)
            for i in range(len(song_group) - 1):
                pairs.append((song_group[i], song_group[i + 1]))
        else:
            pairs.append((song_group[0], song_group[0]))  # Non-cover pair with itself
    return pairs

# Create pairs of cover songs (pair up songs within the same group)
cover_pairs = create_pairs(cover_songs)

# Create pairs of non-cover songs (pair up songs within the same group or with covers)
non_cover_pairs = create_pairs(non_cover_songs)

# Now, we need to ensure we have 1000 pairs for each
selected_cover_pairs = cover_pairs[:1000]
selected_non_cover_pairs = non_cover_pairs[:100]

# Combine and save
final_subset = selected_cover_pairs + selected_non_cover_pairs

# Save to a new JSON file
with open("input.json", "w") as f:
    json.dump(final_subset, f, indent=4)

print(f"Saved {len(final_subset)} pairs to 'subset_1000_cover_pairs_1000_non_pairs.json'")