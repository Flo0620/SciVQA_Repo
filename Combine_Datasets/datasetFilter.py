import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json1_path", type=str, default=None)
parser.add_argument("--json2_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()
json1_path = args.json1_path
json2_path = args.json2_path
output_path = args.output_path

# Load both JSON files
with open(json1_path, "r") as f1:
    data1 = json.load(f1)

with open(json2_path, "r") as f2:
    data2 = json.load(f2)

# Collect all image values from the second JSON
image_files_2 = set(entry["image"] for entry in data2)

# Filter entries from the first JSON
filtered_entries = [entry for entry in data1 if entry["image"] not in image_files_2]

# Write the filtered entries to a new JSON file
with open(output_path, "w") as fout:
    json.dump(filtered_entries, fout, indent=2)