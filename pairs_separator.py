import json

# Function to load, process and save the filtered data
def filter_and_save_json(input_file, output_file):
    # Load the JSON data from the file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    unique_entries = []
    seen_utt = set()

    for pair in data:
        for item in pair:
            if item['utt'] not in seen_utt:
                seen_utt.add(item['utt'])
                unique_entries.append(item)
    
# Save the result to a JSON file
    with open(output_file, 'w') as f:
        json.dump(unique_entries, f, indent=4)

# Define your input and output file paths
input_file = 'input.json'



output_file = 'unique_songs.json'

# Call the function to process the data
filter_and_save_json(input_file, output_file)