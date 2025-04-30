import os
import json

def filter_metadata(metadata_path, output_path=None):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Original metadata size: {len(metadata)} entries")

    # Filter only entries where wav file exists
    filtered_metadata = []
    for item in metadata:
        wav_path = item["wav"]
        if os.path.exists(wav_path):
            filtered_metadata.append(item)
        else:
            print(f"Missing file: {wav_path}")

    print(f"Filtered metadata size: {len(filtered_metadata)} entries")

    # Save to a new file if needed
    if output_path:
        with open(output_path, "w") as f:
            json.dump(filtered_metadata, f, indent=4)
        print(f"Filtered metadata saved to {output_path}")

    return filtered_metadata

filtered_metadata = filter_metadata("datasets/shs100k_unique.json", "datasets/shs100k_unique_filtererd.json")