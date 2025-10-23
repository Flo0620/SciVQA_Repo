import json

# Paths to input and output files
json1_path = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS7.json"
json2_path = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS6.json"
output_path = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS7_minus_6.json"

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