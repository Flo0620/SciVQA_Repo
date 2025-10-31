import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

# Load the JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Filter entries that have the key 'answer'
filtered_data = []
removed_count = 0

for entry in data:
    if 'answer' in entry:
        filtered_data.append(entry)
    else:
        removed_count += 1

# Save the filtered data
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)

# Print the number of removed entries
print(f"Number of removed entries: {removed_count}")