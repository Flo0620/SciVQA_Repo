import json

# Input and output file paths
input_file = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS6ProcessedForHyperparametertuning_train_split.json"
output_file = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS6ProcessedForHyperparametertuning_train_split_filtered.json"

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